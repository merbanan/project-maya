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
| DSv4 MTP | evaluated, not built | RAM-tier verify (upstream #157 pattern); FreeToken/ox-boost measured MTP/DSpark -13% / slower on PCIe-bound offload |
| Hy3 long-context attention | done | code-pair LUT (1d72f3b) + K fragments in registers (2f13857): 36.1 -> 33.5 ms at 66K |
| Hy3 attention, V side in registers | done, opt-in | STRATA_GLM_GQA_REG=2 (cffeaaf): correct, 3.6x slower (60.3 vs 16.8 ms) - off |
| Gemma head-512 prompt attention | done | 2,693 -> 3,826 (26K), 1,106 -> 1,993 (94K) tok/s, needles 3/3 (196cc14, 9a93286) |
| GLM route (1Cat #1147 fused router top-k + quant) | done | router + shared gate/up one multi-GEMV, route + shared SwiGLU + Q8 one kernel; table reads parallel: moe_route 1.61-1.74 -> 1.52 ms (cb48441 / fd32e62) |
| 1Cat mHC fusion (#1172/1166/1159) | present | hc3 grid-barrier kernel (one launch a step) |
| 1Cat TC MLA prefill (#1125) | present | WMMA MLA prompt attention on by default |
| 1Cat verify-row reuse (#1142/1157) | n/a | no GLM MTP on this box |
| project-maya #92 / #131 / #132 / #118 | done | picked onto hy3 / dsv4 / qwen38; #118 opt-in |
| Expert prefetch / fetch overlap (FreeToken, ox-boost) | done, opt-in | STRATA_GLM_PREFETCH_N / _RANK / _AT, STRATA_GLM_FETCH_OVERLAP / _PRIO: all slower here (Hy3 20.2-21.8 vs 22.5 tok/s) |
| FreeToken CPU-vs-PCIe decision per miss | present | CPU lane calibration (0.13 ms CPU vs 0.55 ms PCIe an expert) |
| FreeToken double-buffered prefill expert streaming | present | prestage (160 experts) + disk landing ring |
| FreeToken global expert cache | present, measured | global slot split from routing counts vs uniform per-layer: 22.86 / 21.77 vs 21.14 / 21.22 tok/s |
| FreeToken elastic expert / KV VRAM | present | prompt + KV loan from the expert pool |
| freetoken-ox-boost review | done | its wins (prefetch, soft cache, overlap, FP8) map to the above; FP8 n/a on V100 |
| VRAM release on stop | done | engine exits when its server is gone (e6beee4 ...), and at once after QUIT |

## Serving / tooling

| Item | Status | Evidence |
|---|---|---|
| Tokenizers: llama4 (Muse), gemma4 (Gemma) | done | 13/13 texts = llama.cpp (tests/q35/tokcheck.py); O(n log n) BPE for long pieces |
| Benchmarks in BENCHMARK.md | done | every model, contexts, drafters |
