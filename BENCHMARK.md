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

## Gemma 4 E4B - unsloth Q4_K_M, `strata-dense` (2026-10-14)

Greedy tokens match llama.cpp (128 / 128 at a short prompt; at 6-28K the first difference is a near-tie, llama
margin 0.06-0.33). Assistant drafts: `STRATA_DENSE_MTP=mtp-gemma-4-E4B-it.gguf` (the 4-layer gemma4-assistant on the
target's K/V), `STRATA_DENSE_DRAFT=K`.

| Setting | Prompt (tok/s) | Decode (tok/s) | Notes |
|---|---:|---:|---|
| llama.cpp | 287 (32) / 3,449 (1.5K) / 3,676 (5.6K) / 3,140 (26K) | 92.5 / 92.5 / 91.1 / 85.2 | |
| no drafts | 579 (32) / 4,303 (1.5K) / 4,668 (5.6K) / 2,695 (26K) | **98.8 - 100.1** / 85.6 / 80.3 / **85.3** (26K) | |
| assistant, 2 drafts | | **161.3** | 2.37 tokens a round |
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
