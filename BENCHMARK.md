# Benchmarks (one Tesla V100-PCIE 32 GB)

The models this fork runs, measured on one machine: a **Tesla V100-PCIE 32 GB** (sm_70, HBM2 ~900 GB/s), 2x
Xeon, 62 GB RAM, one NVMe. Greedy decoding unless a row says otherwise; "decode" counts committed tokens a second
over the answer, "prompt" the prompt's tokens a second. llama.cpp is the same GGUF in the llama.cpp server on the
same GPU (`-ngl 99 -fa on`), as a reference.

**Heat:** this V100 throttles under sustained load (81-83 C, SM clock 1380 -> 600-900 MHz). Numbers taken after a
long prefill (the long-context rows) run at a lower clock than the short ones; compare within a row.

Engines: `strata-dense` (branch `qwen38`) for the dense / small-MoE Qwen3.5-architecture models, the GLM fast path
(`strata --glm-pack`) for GLM-5.3-Flash, Hy3 and DeepSeek-V4 Flash (branches `hy3`, `dsv4`).

## Qwen3.8-27B (dense GDN hybrid) - unsloth Q4_K_M, `strata-dense`

| Setting | Prompt (tok/s) | Decode (tok/s) | Notes |
|---|---:|---:|---|
| llama.cpp | 681 (1.6K) | 31.4 | |
| no drafts | 1,017 (1.6K) / 1,063 (12K, chunk 2048) | 33.6 | dp4a GEMV on the packed layout for 1-3 rows (was 29.1 with the tensor-core form) |
| MTP, 3 drafts | | 56.9 (128 tokens) | 2.46 tokens a round (47.9 before the dp4a draft steps) |
| MTP, 7 drafts | | 48.7 | 3.20 tokens a round |
| **DFlash2, 7 drafts** | | **96.6** (code answer) / 79.9 (after a 1.6K prompt) | 4.47 / 3.75 tokens a round; 69.4 at 1.6K before the split-K drafter attention |
| DFlash2, sampled (T 0.6, top-k 20) | | 73.9 | 3.46 tokens a round |
| DFlash2 through `serve/server.py` (T 0.6) | | 70-90 | |

Long context (needles 3/3 at every length):

| Context | KV | Prompt (tok/s) | Decode, no drafts | Decode, DFlash2 |
|---|---|---:|---:|---:|
| 30K | FP16 | 931 | 28.5 | 75.5 |
| 60K | FP16 | 747 | 13.6 | 49.5 |
| 94K | FP16 | 502 | 9.1 | 34.3 |
| 28-30K | K8V4 | 811-916 | 26.3-28.2 | |

A decode step at 90K: 106 ms FP16, 65 ms K8V4 (the K/V read is ~2.5x smaller). Prefix checkpoints: the second turn
of a 19.7K-token conversation whose history drops the first answer's reasoning reads 25 tokens (0.35 s) instead of
19.7K (20.2 s).

Verify window cost (1K context): 1 token 37.6 ms, 8 tokens 40.6 ms (`STRATA_DENSE_BENCH`).

## Qwen3.6-35B-A3B (MoE GDN hybrid) - unsloth UD-Q4_K_M, `strata-dense`

| Setting | Prompt (tok/s) | Decode (tok/s) | Notes |
|---|---:|---:|---|
| llama.cpp | 731 (1.6K) | 88.6 | |
| no drafts | 2,333 (1.6K) / 2,075 (28K) / 1,443 (94K) | **97.8** (short; was 79.5) / 70.3 (28K) / 57.5 (94K) | dp4a experts / packed GEMVs at 1-3 rows, warp router, gate + up one launch; needles 3/3 at 28K and 94K |
| MTP, 3 drafts (unsloth `-MTP-GGUF`) | | 139.2 (256 tokens) / 137.3 (128) | 2.78 / 2.65 tokens a round |
| MTP, 5 drafts | | 135.7 | 3.31 tokens a round |
| DFlash2, 7 drafts (incoai/Qwen3.6-35B-A3B-DFlash2) | | 193.8 (code answer) / 197.1 (after 1.6K) / 162.7 (after 28K) | 4.83 / 5.00 / 4.49 tokens a round |
| **DSpark, 7 drafts** (LithosAI NVFP4 -> F16 GGUF, Q8_0 at load) | | 201.1 (code answer) / **200.4** (after 1.6K) / 162.7 (after 28K, 8K window) | 4.92 / 5.22 / 4.74 tokens a round; needles 3/3 at 28K |

Drafter changes behind these rows (commit e2a748a): split-K drafter attention (a kv head's rows together a chunk
of the context: the 1.6K draft pass 10.4 -> 4.0 ms DFlash2, 18.1 -> 4.8 ms DSpark) and drafter K/V buffers of
their own (the 35B's target K rows are 512 wide, the drafters' 1024: prompt chunks over 1024 rows overwrote the
drafter context, now 4.64 -> 5.00 tokens a round at 1.6K). Before: DFlash2 151.9 / DSpark 131.3 after 1.6K.

### Qwen3.6-35B-A3B speculative decoding, fresh run (2026-10-14, after e98df5a; one session, 256 tokens)

Decode tok/s (tokens a round). "code" = the 34-token code prompt; 1.6K / 28K = the same question after that much
filler (chunk 2048). Greedy unless noted.

| Mode | code | after 1.6K | after 28K | code, sampled T 0.6 top-k 20 |
|---|---:|---:|---:|---:|
| no drafts | 99.0 | 99.6 | 85.9 | 97.6 |
| MTP, 3 drafts | 147.6 (2.81) | 158.0 (3.01) | 153.7 (3.20) | 142.6 (2.71) |
| MTP, 5 drafts | 141.2 (3.28) | 157.1 (3.66) | **166.3** (4.20) | |
| DFlash2, 7 drafts | **209.3** (5.02) | 210.2 (5.12) | 162.6 (4.34) | **165.3** (4.00) |
| DSpark, 7 drafts (8K window) | 201.0 (4.92) | **213.9** (5.33) | 168.2 (4.74) | 156.4 (3.94) |

At 28K the block drafters' extra draft pass costs about what their longer runs gain: MTP with 5 drafts ties them.

Adaptive MTP draft count (default for MoE models: from 5, one more after a full round, back to kept + 1 after a
miss; STRATA_DENSE_ADAPTIVE): 35B code answer K5 141.2 -> 168.5, K7 124.6 -> 172.2 tok/s; after 1.6K K5 157.1 -> 177.4.
(On the dense 27B it lost 2-3%: off there.)

Sampled MTP drafts with the exact min(1, p / q) test (default for MoE models, STRATA_DENSE_SPEC_SAMPLE): 35B code
answer, top-k 20 / top-p 0.95: T 0.6 148.7 -> 155.5 tok/s (2.56 -> 2.75 a round), T 1.0 130.1 -> 147.7 (2.17 -> 2.56).
(27B: -2 to -3%, off there.)

## Gemma 4 E4B - unsloth Q4_K_M, `strata-dense` (2026-10-14)

Greedy tokens match llama.cpp (128 / 128 at a short prompt; at 6-28K the first difference is a near-tie, llama
margin 0.06-0.33). Assistant drafts: `STRATA_DENSE_MTP=mtp-gemma-4-E4B-it.gguf` (the 4-layer gemma4-assistant on the
target's K/V), `STRATA_DENSE_DRAFT=K`.

| Setting | Prompt (tok/s) | Decode (tok/s) | Notes |
|---|---:|---:|---|
| llama.cpp | 287 (32) / 3,449 (1.5K) / 3,676 (5.6K) / 3,140 (26K) | 92.5 / 92.5 / 91.1 / 85.2 | |
| no drafts | 579 (32) / 4,303 (1.5K) / **5,104** (5.6K) / **3,765** (26K; 2,695 before the one-pass head-512 prompt attention) | **98.8 - 100.1** / 85.6 / 80.3 / **85.3** (26K) | |
| assistant, 2 drafts | | **161.3** | 2.37 tokens a round |

Needles 3/3 at 26K / 57K / 94K (tests/q35/q38needle.py --style gemma): prompt 3,826 / 2,730 / 1,993 tok/s with the
one-pass head-512 attention (2,693 / 1,650 / 1,106 with the split form), decode 87.5 / 78.5 / 70.1 tok/s.
| assistant, 3 drafts | | 152.0 / 150.3 (after 1.5K) | 2.69 / 2.64 |
| assistant, 4 drafts | | 157.6 | 2.91 |

## Muse Glimmer 30B - unsloth UD-Q4_K_XL (15.9 GB), `strata-dense` (2026-10-14)

Dense, 52 layers (3 sliding-window + 1 global, NoPE), gated attention 32 / 2 heads of 128. Greedy tokens follow
llama.cpp until a near-tie (llama margin 0.03-0.29). DFlash: the bundled `dflash-kquant.gguf` (5 layers, block 16;
run with 8 rows), `STRATA_DENSE_DFLASH=dflash-kquant.gguf`.

| Setting | Prompt (tok/s) | Decode (tok/s) | Notes |
|---|---:|---:|---|
| llama.cpp | 166 (29) / 774 (1.4K) / 826 (11K) | 37.4 / 37.2 / 36.8 | |
| no drafts | 134 (29) / 981-1,008 (1.4K) / 1,156 (11K) | 34.8 / 33.5-35.1 (1.4K) | |
| **DFlash, 7 drafts, 8 rows** | | **82.4** / 84.7 (after 1.4K) | 3.36 / 3.54 tokens a round, draft pass 5.1 ms |
| DFlash, 7 drafts, 16 rows | | 63.5 | 3.54 tokens a round, draft pass 20.5 ms (the GEMM path) |

## DeepSeek-V4 Flash - antirez IQ2_XXS (80.8 GB), GLM fast path (`dsv4`)

Fresh run 2026-10-14 (dsv4 6d39cfc: tensor-core prompt indexer and attention, parallel compressors), needles 3/3
at every length; prompt = the engine's prompt ms/token, decode = its decode stat over 100 tokens.

| Context | Prompt (tok/s) | Decode (tok/s) | Before (prompt / decode) |
|---|---:|---:|---|
| 2K | 435 | 29.1 | 29.8 cold / 46-49 warm short |
| 28K | **770** | 25.0 | 500 / 22.3 |
| 65K | **714** | 21.7 | ~370 / 20.7 |
| 93K | **625** | 21.7 | 312 / 19.9 (14.5 before the indexer top-k fix) |

A 26K prompt's GPU time: 50.5 s -> 34.9 s (indexer 11.2 -> 2.1 s, attention 12.7 -> 6.0 s). Decode is bound by
the RAM-tier experts (CPU lane ~14.5 ms, fetches) at every length.

MTP (dsv4 0caeb94, `STRATA_DSV4_MTP=<the MTP GGUF>`, tools/d4mtp.py: the block drafts beside every greedy step, 3
prompts x 256 tokens): **83.5%** of drafts match the trunk, 1.81 ms a draft with its experts in VRAM
(`STRATA_GLM_VRAM_GB=27`; 19.6 ms staged), plain decode 29.6-30.9 tok/s, 27.5-28.2 with the shadow draft.
Verify cost (tools/d4verify.py, 2K context, decode step 31.8-33.9 ms), 1 / 2 / 4 / 8 drafts:
token path (one step a token) 1.95 / 2.75 / 4.44 / 8.05 steps; batched prompt path (`STRATA_GLM_PREFILL_MIN=1`)
4.61 / 4.60 / 4.78 / 5.53 steps - ~150 ms fixed (pool lend, plan, PCIe staging of RAM-tier experts), ~4.5 ms a row.
At 83.5% the expected tokens a round are 1.84 / 2.53 / 3.6 / 4.9: 0.38-0.87x on either path, so speculation is off.
A gain needs a multi-row token path (dense weights read once, the CPU lane over both rows): ~1.2x at best by the
decode split (dense ~9 ms, experts ~23 ms a token).

MTP speculative decode (dsv4 5a6bddc, `STRATA_DSV4_MTP=<GGUF>`; two rows a round, dense GEMVs shared by mv2,
tokens = plain decode's with fixed tiers): 87-89% of drafts accepted, a round 55-59 ms (verify 51-55, drafts 3.1-3.6)
for ~1.88 tokens. tools/d4mtp.py, 3 prompts x 256 tokens:

| Setting | Plain (tok/s) | MTP spec (tok/s) |
|---|---:|---:|
| equal pools (`STRATA_GLM_VRAM_GB=27`) | 29.7 / 29.7 / 30.5 | **31.6 / 31.9 / 32.7** |
| default pool, MTP experts Q4_K (3.81 GB kept out) | 31.4 / 31.4 / 32.6 | 32.1 / 31.6 / 33.3 |
| default pool, MTP experts q3_k (2.95 GB) | 31.0 / 31.9 / 33.2 | 32.1 / 30.3 / 33.9 |
| default pool, MTP experts q2_k (2.30 GB) | 31.7 / 31.7 / 33.0 | 32.9 / 30.0 / 34.1 |
| 28K prompts | 26.3 / 26.3 / 26.3 | 26.5 / 24.2 / 25.8 |

Split verify (dsv4 c193e7f, the default; after lighttransport/Strata glm53f and PolyStrata): the rows half a
layer apart, one row's CPU-lane experts under the other row's attention. Same run, default pool: plain 31.3 / 31.6 /
33.2, split **34.4 / 33.0 / 34.7** (+6%, a round 54.0 ms), joint (`STRATA_DSV4_SPEC_JOINT=1`) 32.8 / 31.6 / 33.4.
28K: plain 26.2 / 26.4 / 26.8, split 28.0 / 24.9 / 24.9 (a round 74.7 ms), needles 3/3.

Round profile: dense GEMVs 10.3 ms (one row 10.0), MoE side 34.5 ms (one row 17.6: CPU-lane wait 14.3, fetch 6.1) -
the expert pipeline is CPU-lane bound and costs both rows in full, so the gain is the shared dense reads minus the
VRAM the MTP block takes from the pool: even at default settings.

DSv4 decode profile at 2K (35.0 ms a token): CPU-lane wait 8.3, grouped output GEMVs 5.1, attention 3.6, expert
fetch 3.5, q_b 3.2, expert gate/up + down 4.2, router + shared 1.8, projections 1.7, route 1.5, mHC 1.6. A Q8_0 GEMV
with 2 / 4 / 8 loads a lane in flight: 27.5 / 27.5 / 26.5 tok/s vs 28.4-28.6 - slower, not kept.

## GLM-5.3-Flash - Maya-S24 (94.7 GB), live Maya (v1.0.30 + PR #78)

| Setting | Prompt (tok/s) | Decode (tok/s) | Notes |
|---|---:|---:|---|
| v1.0.30 / v1.1.3 | ~290 (2K) / ~585 (8K) | 26-27 | |
| `STRATA_GLM_TRUNK_KEEP=7` | | 28.7 | KL 0.0136 (changes the answers) |
| at 60K, trunk keep map | | 28.4 (vs 24.4) | KL 0.0175 |

## Hy3 (Hunyuan 3) - `hy3`

| Context | Decode (tok/s) | Notes |
|---|---:|---|
| 93K | 7.84 | non-streamed KV |
| 93K, KV streaming (16K / 32K cache) | 10.15 / 10.41 | disk reads 21-24 a token from RAM pressure |
| 28K tq6v3 (2026-10-14) | 15.47 | attention 16.9 ms a token (17.9 before the code-pair tables + register K fragments); 15.12 before |
| 66K tq6v3 (2026-10-14) | 11.39 | attention 33.5 ms a token (36.1 before) |

Expert prefetch (`STRATA_GLM_PREFETCH_N` 1 / 2) and fetch overlap (#118) are slower on this PCIe-bound box: 16.4 / 15.6
and 20.1 tok/s against 22.9 at a short context.
