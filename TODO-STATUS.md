# TODO list - status (2026-10-14)

Every item from the working TODO list, what was done, the measurement (one Tesla V100-PCIE 32 GB), and where it
lives. Branches on the fork: `qwen38` (strata-dense: Qwen3.8-27B, Qwen3.6-35B-A3B, Gemma 4 E4B, Muse Glimmer),
`dsv4` (DeepSeek-V4 Flash), `hy3` (Hy3 / GLM fast path). Numbers: BENCHMARK.md.

## Ports

| Item | Status | Evidence |
|---|---|---|
| Qwen3.8-27B (MTP, DFlash2, K8V4, checkpoints) | done | greedy = llama.cpp; DFlash2 96.6 tok/s; needles 3/3 to 94K |
| Qwen3.6-35B-A3B (MTP, DFlash2, DSpark) | done | decode 97.8 (llama.cpp 88.6), DSpark 214 / DFlash2 210 tok/s after 1.6K |
| DeepSeek-V4 Flash | done | prompt 770 / 625 tok/s at 28K / 93K, needles 3/3 |
| Gemma 4 E4B (+ assistant drafter, serving) | done | decode 99-100 (llama.cpp 92), assistant 161 tok/s, needles 3/3 to 94K, served ~170 tok/s |
| Muse Glimmer 30B (+ DFlash, serving, tool calls) | done | DFlash 82 tok/s (35 plain), served 94 tok/s, tool call + result turn tested live |

## Speed items

| Item | Status | Evidence / commit |
|---|---|---|
| 35B plain decode below llama.cpp | done | dp4a GEMVs at 1-3 rows, warp router, gate+up launch: 79.5 -> 97.8 tok/s (e98df5a) |
| 27B T=1 regression from packing | done | packed dp4a + Q4_K/Q5_K min-term fix: 33.6 tok/s, 128/128 greedy (e98df5a, e9815d9) |
| Drafter attention at long context | done | split-K: 35B DFlash2 1.6K 151.9 -> 182.9, DSpark 131.3 -> 185.7 (e2a748a) |
| CUDA graphs / GPU-resident rounds | done, opt-in | segment graphs measured: no gain (STRATA_DENSE_GRAPH=1) |
| Adaptive K (MTP) | done | default for MoE: 35B MTP K5 141 -> 168, K7 125 -> 172 tok/s; off for dense (27B -3%) |
| Sampled-draft p/q acceptance (lithos) | done | MTP drafts sampled from q, exact min(1,p/q) + residual: 35B T0.6 148.7 -> 155.5, T1.0 130.1 -> 147.7 tok/s; default for MoE |
| DSv4 prompt indexer on tensor cores | done | d4_index 11.2 -> 2.1 s, d4_attn 12.7 -> 6.0 s at 26K (6d39cfc) |
| DSv4 MTP | done, faster at short context | MTP block + speculative decode (dsv4 0caeb94, 5a6bddc, c193e7f; on with STRATA_DSV4_MTP): split verify (rows half a layer apart) 34.4 / 33.0 / 34.7 vs plain 31.3 / 31.6 / 33.2 tok/s (+6%), 87% accepted, tokens = plain decode's; 28K even (24.9-28.0 vs 26.2-26.8), needles 3/3 |
| Hy3 long-context attention | done | code-pair LUT (1d72f3b) + K fragments in registers (2f13857): 36.1 -> 33.5 ms at 66K |
| Hy3 attention, V side in registers | done, opt-in | STRATA_GLM_GQA_REG=2 (cffeaaf): correct, 3.6x slower (60.3 vs 16.8 ms) - off |
| Gemma head-512 prompt attention | done | 2,693 -> 3,826 (26K), 1,106 -> 1,993 (94K) tok/s, needles 3/3 (196cc14, 9a93286) |
| DSv4 / GLM prompt: shared expert sub-batch 2048 | done | DSv4 26K prompt GPU time -1.9% (shexp 1.73 -> 0.90 s), Hy3 neutral, needles 3/3 (dsv4 ed8c78d, hy3 279f798) |
| GLM route (1Cat #1147 fused router top-k + quant) | done | router + shared gate/up one multi-GEMV, route + shared SwiGLU + Q8 one kernel; table reads parallel: moe_route 1.61-1.74 -> 1.52 ms (cb48441 / fd32e62) |
| 1Cat mHC fusion (#1172/1166/1159) | present, measured | hc3 fused kernel vs the old form (STRATA_GLM_HC1=1), DSv4 2K: hc_attn + hc_ffn 1.60 vs 2.96 ms a token, decode 28.37 / 27.74 vs 27.65 / 27.07 tok/s, needles 3/3 |
| 1Cat TC MLA prefill (#1125) | present | WMMA MLA prompt attention on by default |
| 1Cat verify-row reuse (#1142/1157) | done (DSv4) | the DSv4 verify keeps an accepted row's state (no recompute); a rejected row only has its compressor windows put back (5a6bddc) |
| project-maya #92 / #131 / #132 / #118 | done | picked onto hy3 / dsv4 / qwen38; #118 opt-in |
| Expert prefetch / fetch overlap (FreeToken, ox-boost) | done, opt-in | STRATA_GLM_PREFETCH_N / _RANK / _AT, STRATA_GLM_FETCH_OVERLAP / _PRIO: all slower here (Hy3 20.2-21.8 vs 22.5 tok/s) |
| FreeToken CPU-vs-PCIe decision per miss | present | CPU lane calibration (0.13 ms CPU vs 0.55 ms PCIe an expert) |
| FreeToken double-buffered prefill expert streaming | present | prestage (160 experts) + disk landing ring |
| FreeToken global expert cache | present, measured | global slot split from routing counts vs uniform per-layer: 22.86 / 21.77 vs 21.14 / 21.22 tok/s |
| FreeToken elastic expert / KV VRAM | present | prompt + KV loan from the expert pool |
| freetoken-ox-boost review | done | its wins (prefetch, soft cache, overlap, FP8) map to the above; FP8 n/a on V100 |
| VRAM release on stop | done | engine exits when its server is gone (e6beee4 ...), and at once after QUIT |

## Open

| Item | Status | Evidence |
|---|---|---|
| Forks in Niko1221/Strata#2067 (lighttransport/Strata `glm53f`, NenadSteric/PolyStrata; iMilnb's HIP work is inside PolyStrata; VecSzn's repo is gone) | reviewed, applied | split verify -> DSv4 MTP +6% (c193e7f); GLM DFlash2 on the V100 (glm-dflash); GLM prompt attention -46% (a4f2f49). Already in Maya: V100 codebook-in-shared expert GEMVs, device-written route ring (the GPU mailbox), prestaged prefill experts, VRAM promotions / swaps (the adaptive cache). KDA prefill: the recurrence split over 4 blocks a head (48f2e0f). Lossy ones built opt-in and measured (glm-dflash 29ca355): STRATA_GLM_ROUTE_AFFINITY 0.02 / 0.08 +7% / +21% decode, STRATA_GLM_DEFER_CPU +15% - both change the greedy tokens within the first few, so off by default |

## GLM-5.3-Flash DFlash2 (branch `glm-dflash` = upstream v1.1.10 + fixes)

| Item | Status | Evidence |
|---|---|---|
| DFlash2 drafter for GLM | done (upstream had it: v1.1.10 `--dflash`), made to run and gain on one V100 | 647b213, 7de22a5 |
| V100 launch OOM (window grouping kernel's 2.6 KB stack = ~420 MB reserved) | fixed | runs at the live 400 MB reserve |
| Window's RAM-tier experts over PCIe on a 1-NUMA box | fixed: CPU lanes per L3 domain (two CCDs), default | 2 drafts: 21.5/20.1/25.0 -> 24.4/23.9/28.0 tok/s |
| VRAM tier froze under the window's all-CPU plan | fixed: background swaps run beside it | hit rate 41-45% -> 51-60% |
| Result, short prompts (defaults, Q4_K_M drafter) | +3-5% | plain 26.7 / 27.1, DFlash2 27.6 / 28.5 tok/s avg (two rounds), 95% accepted |
| Result, 12K prompts | -4% | plain 22.6 / 25.2 / 24.9, DFlash2 21.8 / 22.3 / 25.4 |
| GLM KDA prompt recurrence (fork review: lighttransport's KDA prefill kernels) | done, default | 4 blocks a head instead of 1 (kda_rec_split): kda_rec 4.7 / 5.1 -> 3.5 s at 27K, ~2% of the prompt (inside its run spread); rel L2 1e-7 (glm-dflash 48f2e0f, hy3 / dsv4) |
| GLM prompt attention on Volta (fork review follow-up: the DSA attention was 28.7% of a 27K prompt) | done, default | two-pass 32-head kernel: dsa_attn 22.5 / 23.2 -> 12.2 / 12.5 s, 27K prompt GPU time -10% / -12% (glm-dflash a4f2f49; hy3 9927b78: -6%, dsv4 c796d98); rel L2 vs F32 unchanged |

## Open

| Item | Status | Evidence |
|---|---|---|
| Gemma 4 30B-class model support (strata-dense) | in progress | |
| salient-data/qwen38-next-dgx-recipe: patches that could speed up prefill / decode | reviewed: little applies | vLLM recipe for Qwen3.8-Flash-Next NVFP4 on a DGX Spark (67.5 tok/s 1 stream). Patches: PLE host-file gather (that model's 51 GB n-gram table; none of ours), FP8 dense cast (no FP8 on V100), NVFP4 draft head / sm12x alignment (Blackwell), mamba block-table fuse and GLM kpool tail (vLLM internals). The one idea that carries: a reduced-vocabulary draft head - strata-dense has one already; GLM DFlash2's head scores the full 154K vocab (2.1 ms a round, ~2% of it) |
| Upstream PRs (mw00/project-maya), rebased on main 8931339 (v1.1.11), measured before / after on the V100 (raw logs: prrun.log) | sent, updated | base: DFlash fails (84 launches), plain 24.4/27.8/27.2, 27K prompt 78.4 s. #195 windows run (DFlash 19.9/24.7/26.7); #196 prompt 67.9 s (dsa_attn 22.5 -> 11.8 s); #197 with #195 DFlash 26.0/27.6/32.2 (plain 25.6/28.9/28.6); #198 VRAM hit 47.6/60.8/53.9 -> 53.1/66.6/58.9% (speed within noise); #199 kda_rec 4.71 -> 3.31 s (prompt 78.0 s); plain decode unchanged by each |
| Shared-expert sub-batch 2048 on GLM | measured on 8931339: not faster, no PR | STRATA_GLM_PREFILL_SUB unset (the default it changes): 27K prompts base 309.9 / 316.6 s (shexp 31.2 / 32.4 s), with it 315.2 / 317.3 s (31.8 / 32.5 s); with 1024: shexp 1,617 vs 1,618 ms |
| GLM prompt path with STRATA_GLM_PREFILL_SUB unset (upstream's NVIDIA default) | open - found | 27K prompts: 314 s of GPU time against 78.6 s with 1024 (shared expert 32 s vs 1.6 s, dsa_attn 30 vs 22.5 s) on the V100 |
| Strata (Niko1221) tree: update, then a V100-optimized tree from Strata's PRs, benchmarked | done | merbanan/Strata `main` = upstream 296b59d (old head `main-2026-10-01`). `v100` 840624c, both of Strata's engines measured on the V100: strata-q36 (Qwen3.6-35B-A3B) - grouped decode attention + float4 key loads: 16K 56.4 -> 94.3 tok/s, short 99.6 -> 103.8, same tokens; Flash-Next (Qwen3.8-Flash-Next GSQ-RCO Q2_0, downloaded for this) - Strata #2357 + #2358 on by default on sm_70: decode 56.0-56.2 -> 56.8-56.9 avg (+1.3-2%), same tokens, prompt 729 tok/s unchanged; expert mode 4 (#2069's layout) default on sm_70 (35c9285): 57.5 vs 56.8 / 57.2 avg (+1.2%), now 58.2 / 57.8 / 56.7 tok/s, same tokens. Measured and not taken (noise on Flash-Next): #2221, #2091, #2222, #1949, #2447's STRATA_NO_MMVQ_ROWS2; #1822 / #2081 conflict with main (no port). bench/v100: q36_bench.sh, fn_bench.py, kernel_suite.sh (20 parity / bench tools, all of the strata-q36 kernels pass; gemm_f32_parity's cuBLAS bit-compare differs on sm_70), README, STRATA_PRS.md (400 PRs; the 27 that change strata-q36's kernel files each listed; #1909, #2058, expert modes, the V100 table, #2357, #2358 measured; #2356 n/a: Q2_0 experts) |
| Niko1221/Strata code, issues and PRs: ideas for this work | done | 400 PRs / 600 issues filtered by title, the CUDA / V100 ones read, upstream built and measured on this V100 - below |
| - #2464: Qwen3.6-35B-A3B on a V100 (upstream q36 111 tok/s plain on UD-Q4_K_XL, 56-59 at 16K) | measured: ours is faster | upstream strata-q36 built here (Niko1221/Strata main 296b59d, sm_70), our UD-Q4_K_M file: tg256 94.3 tok/s, pp512 1,685 tok/s; strata-dense 97.8 tok/s decode, 2,333 tok/s prompt (1.6K), 70.3 at 28K vs their 56-59 at 16K |
| - native_mmvq V100 row table (STRATA_SM70_TABLE, PR 1401) | measured: no gain | upstream strata-q36 on our V100-PCIe: 94.3 -> 93.1 tok/s with it; our dp4a GEMVs are faster already |
| - #1949: CPU AVX2 dot kernels with accumulators in registers | n/a | a fix of Strata's own AVX2 dots; Maya's CPU lane runs ggml-cpu's vec_dot (lighttransport measured those as fast as a custom kernel); in Strata's Flash-Next measured as noise (57.4 / 57.1 / 55.9 tok/s) |
| - #2227: MTP hnorm on the stream | n/a here | Flash-Next's MTP option; GLM's NextN decode needs two GPUs (one GPU drafts with DFlash2), DSv4's MTP already normalises each stream (ds4) |
| - #2357 / #2358: sm_70 argmax_rows (+0.7-0.8%), fused GR norm (+0.5%) | taken in the Strata v100 tree | Flash-Next path (not in our engines): +1.3-2% there, on by default on sm_70 (e584be8) |
| jmnargi/Strata-V100 (V100 fork of Strata, main c79f8e1, 2026-10-14; read only, nothing run) | reviewed, items below | Flash-Next on 1-2 V100-PCIE-16GB, CUDA 12.8. Base merged with upstream 61b3fb5 (10-10), ~387 upstream commits behind our v100 tree's 296b59d. Upstream already has its tiled FP32 QSA scorer (7db7a5d5), #540 / #600 / #627 ports, #783 verify-window work, expert mode 8, the Volta MMVQ table. Fork-only, worth porting: |
| - FP16 tensor-core routed experts for the prompt (src/prefill/moe_fp16tc.cu, 31ad437c; Q2_0 experts, q8_1 activations x FP16 weights, FP32 accumulate, compact expert scratch + 192-slot ring) | todo: port to Strata v100 tree, then the idea to Maya | their Flash-Next Q2_0 prefill +25-35% at 2K-32K (32K 1,812 -> 2,259 tok/s), not bit-exact (rel RMS vs FP64 1.073e-2 vs MMQ's 1.074e-2). Our v100 Flash-Next prompt is 729 tok/s on MMQ dp4a - biggest single item. For Maya / strata-dense the same trade: V100 has no int8 tensor cores, our MoE prefill (GLM, DSv4, Qwen3.6) runs dp4a MMQ - check how much of prompt time is expert GEMM first |
| - WMMA FP16 grouped expert GEMMs for IQ formats (IQ3_XXS / IQ3_S / IQ2_S / IQ4_NL, iq_kernels.cu ~3900, 5583bf61 + pf_wmma_parity) and short-row 64-row tiles for groups <= 128 rows (ca46b7e6, a85389ce: kernel -27..29% gate/up, -9% down) | todo: port, measure | IQ3_S pack dual-V100 prefill +20-41% (with GPU-only KV growth). Relevant for Maya models with IQ experts (Qwen3.6 UD files, GLM IQ packs) and the Strata IQ packs |
| - mma884 prompt-attention latency pipeline: all global loads before stores, double-buffered page-table lookups row[2][CH] (2680b4de) | todo: port to v100 tree; compare with Maya's #196 Volta prompt attention | 2,048 queries of a 32K selection: int8 12.11 -> 10.25 ms, fp16 KV 25.7 -> 17.4 ms; output unchanged. Same load-batching idea may apply to Maya's dsa_attn Volta kernel |
| - agentic short prefill: compact CPU-expert activation rows, kernel-param expert pointer tables (no setup kernel), stage-owned buffers (a64694e3) | low priority | 2K cold -6%, 256-2K additions -3..4%; >4K noise |
| - multi-GPU / host pieces (GPU-only KV growth, CPU-assisted prefill across stages, L2/L3 conversation cache, refill overlap) | n/a here | dual-GPU or serving features; one V100 32 GB here |
| Luce-Org/lucebox (main 6f2b179, 2026-10-15; code, 400 PRs, 93 issues read, nothing run) | reviewed, items below | spec-decode server on patched ggml (R9700 / Strix Halo / 3090). Qwen3.8-27B + DFlash2 208 tok/s on an R9700 (6.4x AR), DSv4 Flash + DSpark 39-42 tok/s on a Strix Halo, Gemma 4 31B DFlash 3.2x. V100 only generic (issue #485: our own report there, DFlash 53 vs AR 25 tok/s on Qwen3.6-27B; acceptance 6.4 vs 8.3 on a 3090, FP16 draft rounding suspected). Already in Maya: DFlash2 selector heads (q35_dflash), DSpark Markov head, drafter split-K, device route mailbox, expert prefetch (measured slower here), route affinity bias (STRATA_GLM_ROUTE_AFFINITY, lossy, opt-in) |
| - GDN fast rollback: exact F32 per-token SSM states written by the verify kernel straight into a persistent buffer, restore = copy state[accept_n] (no snapshot / replay), bonus token becomes draft[0] of the next round (no extra forward) (src/common/chain_rollback_policy.h, #506, gated_delta_net.cu tree_persist) | todo: compare with strata-dense's checkpoints | +29-30% decode (3090 x2, 2080 Ti 36.5 -> 47.7), byte-identical; F16 intermediates +5.5 tok/s more; cost ~1.1-1.65 GiB |
| - GDN verify route fusion: ssm_conv_tree_step (history + conv + silu), mapped_verify (state read in place, no final-state write), raw gates (sigmoid / softplus / dt_bias / A in the kernel) (828ecf14, #707) | todo: strata-dense q38 / q36 verify | 31 launches a layer -> a few, +8.4%, byte-identical |
| - verify width: block 16 on a block-8 DFlash2 drafter, capped at 8 above 8K context (qwen35_backend.cpp ~3160), adaptive 4 / 8 / block controller from per-depth acceptance / round cost (adaptive_spec_width.h, #779, #729 confidence head) | todo: measure 8 / 16 / adaptive on the V100 for q38, q36, GLM | 3090-class: 40K 32.7 (16) vs 45.5 (8) tok/s; truncating a 16 draft accepts better than drafting 8 (0.414 vs 0.375). Our GLM DFlash2 loses 4% at 12K - the long-context cap may be why |
| - SpecLA: next verify applies the last round's accepted factors to the state tile in registers, no commit kernel (docs/SPECLA.md, #606); GDN replay log [g,k,delta] per token instead of S_v^2 states | idea | +8.2% on Qwen3.6-27B; replay log saves memory / bandwidth for wider verify |
| - grouped MUL_MAT_ID for small verify batches: sort (token, slot) by expert so warps sharing an expert hit L1/L2 (#501, 170bef45) | todo: q36 / GLM / DSv4 verify expert GEMVs | bit-exact; +3.7% at width 8, ~+10% at 15; 8 rows read 64 experts, 39 distinct |
| - drafter K/V ring: project only the new target-feature rows each step, one replayed graph; drafter window = its trained 40,960 not 4,096 (#740: accept collapsed past 4K) | check: strata-dense drafter window / KV reuse | ~10 -> 3 ms a draft on a 3090; deep-context acceptance |
| - DSpark for DSv4 Flash (DeepSeek's own block drafter under mtp.*: 5 rows in one pass, Markov head, confidence head drives verify width q2-q5) | todo: check our DSv4 GGUF for mtp.* DSpark tensors | 39 tok/s code / 25 prose on a Strix Halo vs our sequential MTP +6%; lossless (batch-invariant verify) |
| - DSv4 long context: V4.1 candidate blocks (index scores max-pooled per 8 rows, top-2048 blocks, per-row top-k inside; no-op below 16K rows), verifier reads the F16 MLA cache directly, exact block-radix top-512 instead of a sort | todo: DSv4 / GLM DSA indexer | lossless except the sparse verifier (opt-in) |
| - KVFlash `qk` policy: fixed KV pool of 64-token chunks, chunks scored by mean-pooled post-RoPE K against current Q every >=64 tokens, cold chunks paged to host | idea, lossy | Qwen3.6-27B 256K: 38.6 vs 13.1 tok/s, needles 14-15/16 at 6% resident; candidate for Hy3 / DSv4 / GLM long-context decode |
| - next-layer router prefetch: layer L's FFN input through layer L+1's norm + router, start streaming predicted experts a layer early (deepseek4_graph.cpp ~4771) | re-check | 80% prediction accuracy; our FreeToken-style prefetch measured slower, but theirs predicts from the real next router |
| - O_DIRECT bulk expert reads during prefill (no page-cache pollution), demand loads ahead of prefetch, mlocked spill tier (unlocked pages were reclaimed) (moe_hybrid_expert_cache.cpp) | check against Maya's RAM tier | - |
| - PFlash (0.8B scorer compresses the prompt, keep 5-30%), Gemma 4 DFlash drafters (z-lab, block 16), BF16 -> FP16 drafter weights on pre-Ampere | PFlash lossy - low priority; Gemma 4 31B: use z-lab DFlash drafter as an alternative to the assistant drafter, size the SWA ring W + verify width | 128K TTFT 10.4x (single-needle only) |
| - #2356: Volta prompt-expert WMMA kernel for IQ2_S / IQ4_NL | n/a | none of our models' experts are IQ2_S / IQ4_NL (GLM / DSv4: IQ2_XXS + Q2_K, Qwen: Q4_K / Q5_K) |
| - not applicable: the SYCL / HIP / Metal / Arc / gfx PRs, multi-GPU layer splits, setup / serve / MCP changes | - | |

## Serving / tooling

| Item | Status | Evidence |
|---|---|---|
| Tokenizers: llama4 (Muse), gemma4 (Gemma) | done | 13/13 texts = llama.cpp (tests/q35/tokcheck.py); O(n log n) BPE for long pieces |
| Benchmarks in BENCHMARK.md | done | every model, contexts, drafters |
