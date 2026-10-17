__global__ void sampler_legacy_kernel(const float* __restrict__ logits, int n_vocab, int n_tokens,
                               const int* __restrict__ history, int history_len, const SamplerParams p,
                               int* __restrict__ out) {
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const float* l = logits + (size_t) t * n_vocab;

    // Temperature is needed by BOTH stages below, so it is computed here; the chain still APPLIES it after
    // the truncation filters - the survivors are chosen on the raw logits and only then scaled.
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;

    // The penalty window is the last `penalty_last_n` entries of this row's history (disabled at this
    // launch: `sample_tokens` refuses a non-zero `penalty_last_n` without a history buffer).
    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }

    // the membership bitmap, as in `sampler_greedy_kernel` - see the cost note there.  The gate needs an
    // NON-EMPTY WINDOW too: the launch sizes the bitmap only when penalties are on, so a caller that hands over
    // a stale history buffer with `penalty_last_n == 0` must not touch it.
    extern __shared__ unsigned int penal_bits[];
    const int bits_words = (int) ((n_vocab + 31) / 32);
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < bits_words; w += blockDim.x) penal_bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < hlen; i += blockDim.x)
            if (hrow[i] >= 0 && hrow[i] < n_vocab)   // an id outside the vocabulary is never a candidate
                atomicOr(&penal_bits[hrow[i] >> 5], 1u << (hrow[i] & 31));
        __syncthreads();
    }
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };

    // top_k in 1..64 is taken as given; 0 ("off") and anything wider mean the widest shortlist the kernel
    // keeps, 64.  Every row writes out[t]: a verify window reads all of them.
    const int KMAX = 64;
    int k = (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX;
    if (k > n_vocab) k = n_vocab;

    // ---- top_k: k rounds of a block argmax over the not-yet-taken.  `sel_*` holds the kept ids and their
    // raw logits in selection order: descending by value, ties to the lower index, which is the order the
    // top_p cut below is defined over.
    __shared__ int sel_ids[KMAX];
    __shared__ float sel_logit[KMAX];
    __shared__ float sv[32];
    __shared__ int si[32];
    for (int i = 0; i < k; ++i) {
        // `n_vocab` is the "no candidate" index: it loses every comparison to a real one (same convention as
        // the greedy kernel, whose tie rule this reduction shares).
        float bv = __int_as_float(0xff800000);   // -inf
        int best = n_vocab;
        for (int v = threadIdx.x; v < n_vocab; v += blockDim.x) {
            bool taken = false;
            for (int j = 0; j < i; ++j) if (sel_ids[j] == v) { taken = true; break; }
            if (taken) continue;
            const float s = apply_penalties(l[v], hit_count(v), p);
            if (s > bv) { bv = s; best = v; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
            const int oi = __shfl_down_sync(0xFFFFFFFFu, best, off);
            if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
        }
        const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
        if (lane == 0) { sv[warp] = bv; si[warp] = best; }
        __syncthreads();
        if (warp == 0) {
            const int nw = (int) ((blockDim.x + 31) >> 5);
            float wv = lane < nw ? sv[lane] : __int_as_float(0xff800000);
            int wi = lane < nw ? si[lane] : n_vocab;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_down_sync(0xFFFFFFFFu, wv, off);
                const int oi = __shfl_down_sync(0xFFFFFFFFu, wi, off);
                if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
            }
            if (lane == 0) { sel_ids[i] = (wi < n_vocab) ? wi : 0; sel_logit[i] = wv; }
        }
        __syncthreads();
    }

    // ---- top_p over the top_k list (penalised logits, descending as the selection produced them), then min_p,
    // then temperature and one Philox draw - llama.cpp's order (issue #53).  Every thread computes the same chain
    // redundantly over `sel_*` - the arithmetic is the serial kernel's, instruction for instruction - so they
    // agree on `pick` and thread 0 writes it.
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = fmaxf(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += exp((double) sel_logit[i] - (double) mx);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += exp((double) sel_logit[i] - (double) mx) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    // ---- min_p on top_p's survivors: the descending prefix whose probability is at least `min_p` of the top
    // token's.  In logit space the threshold is `sel_logit[0] + logf(min_p)` - equivalent to `p >= min_p * p_max`
    // without the overflow an exp of raw logits risks.  0 disables, and the head itself always survives
    // (`expf(0) == 1 >= min_p` for min_p in 0..1), so the count never reaches zero.
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + logf(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    // temperature only: the penalties were applied once, before the selection (issue #53: they were applied a
    // second time here, after the temperature scaling - llama.cpp's chain has one penalties stage)
    auto scaled = [&](int i) { return sel_logit[i] * inv_t; };
    float smx = scaled(0);
    for (int i = 1; i < n_keep; ++i) smx = fmaxf(smx, scaled(i));
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += exp((double) scaled(i) - (double) smx);
    const float u = philox_uniform(p.seed, p.counter + (uint64_t) t);
    double cum = 0.0;
    int pick = sel_ids[n_keep - 1];
    for (int i = 0; i < n_keep; ++i) {
        cum += exp((double) scaled(i) - (double) smx) / sum;
        if ((double) u < cum) { pick = sel_ids[i]; break; }
    }
    if (threadIdx.x == 0) out[t] = pick;
}
