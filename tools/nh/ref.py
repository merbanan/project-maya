"""tools/nh/ref.py - a whole-sequence Nemotron-H forward in torch over the EXL3 model (maya-nh's oracle).

    python tools/nh/ref.py --model ~/models/Nemotron-3.5-Super-VL-EXL3-4.23bpw --text "The capital of France is" \
        [--dump out.npz] [--gen 8] [--layers 12]
    python tools/nh/ref.py --model ... --ids 1,2,3

It follows transformers' modeling_nemotron_h.py (NemotronHBlock and its mixers, the language model of
NVIDIA-Nemotron-3.5-Super-VL's NemotronH_Omni_Reasoning_V3) from position 0 - every position at once - with each
layer's weights decoded from EXL3 when the layer runs (tools/mm3/exl3_torch.py) and freed after it:

  - each block: h += mixer(rms(h) * w) - plain RMSNorm, eps layer_norm_epsilon; one mixer a block, by
    layers_block_type: "mamba", "attention" or "moe"; the final norm_f, then lm_head.
  - mamba (Mamba-2): in_proj -> z [heads x head_dim], xBC [heads x head_dim + 2 groups x state], dt [heads];
    xBC through the causal depthwise conv (kernel 4, bias) and SiLU, split into x, B, C; dt = softplus(dt + dt_bias)
    (dt_limit (0, inf): no clamp - mamba_ssm's kernels, the model's own path); A = -exp(A_log); per head h (group
    h / (heads / groups)), state S [head_dim][state]: S = exp(dt A) S + dt x B^T, y = S C + D x; then the gated
    RMSNorm (norm_before_gate False): rms over groups of heads x head_dim / groups of y * silu(z), times norm.weight;
    out_proj.
  - attention: GQA (num_attention_heads on num_key_value_heads, head_dim), no position encoding, causal softmax
    attention at scale 1/sqrt(head_dim).
  - moe: router logits x . gate.weight in F32, s = sigmoid, chosen by s + e_score_correction_bias (top
    num_experts_per_tok, one group), weights s over the chosen renormalised (+ 1e-20) times routed_scaling_factor;
    the routed experts on the latent: fc2(sum_i w_i down_i(relu(up_i(fc1(x)))^2)), plus the shared expert
    down(relu(up(x))^2) on x itself.

Activations are F32 throughout; the EXL3 matrices decode exactly.  The Mamba scan is the plain recurrence, token by
token (the chunked SSD of the model's kernels is the same sums in another order).

--dump writes the ids, the logits of the last --dump-last positions, and the residual stream after every layer at
those positions (unless --no-hs), for the engine's parity checks (tools/nh/compare.py).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mm3"))
import exl3_torch as ET  # noqa: E402
from st import SafeTensors  # noqa: E402

PFX = "language_model.backbone."


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def lin(st, prefix: str, x: torch.Tensor, dev) -> torch.Tensor:
    w = ET.weight(st, prefix, dev)
    y = x @ w.T
    del w
    return y


def many(st, prefixes: list[str], dev) -> list[torch.Tensor]:
    if ET.is_exl3(st, prefixes[0]):
        return ET.decode_many(st, prefixes, dev)
    return [ET.weight(st, p, dev) for p in prefixes]


def vec(st, name: str, dev) -> torch.Tensor:
    return torch.from_numpy(np.array(st.f32(name))).to(dev)


def mamba(st, L: str, x: torch.Tensor, c: dict, dev) -> torch.Tensor:
    T = x.shape[0]
    H, P, N, G = c["mamba_num_heads"], c["mamba_head_dim"], c["ssm_state_size"], c["n_groups"]
    d_in = H * P
    conv_dim = d_in + 2 * G * N
    m = L + "mixer."
    zxbcdt = lin(st, m + "in_proj", x, dev)                                  # [T, d_in + conv_dim + H]
    z, xbc, dt = torch.split(zxbcdt, [d_in, conv_dim, H], dim=-1)
    cw = vec(st, m + "conv1d.weight", dev).reshape(conv_dim, -1)            # [conv_dim, kernel]
    cb = vec(st, m + "conv1d.bias", dev)
    kk = cw.shape[1]
    xp = F.pad(xbc.T, (kk - 1, 0))                                          # [conv_dim, T + kk - 1]
    conv = cb[:, None].expand(-1, T).clone()
    for j in range(kk):
        conv += cw[:, j:j + 1] * xp[:, j:j + T]
    xbc = F.silu(conv.T)                                                    # [T, conv_dim]
    xs, B, C = torch.split(xbc, [d_in, G * N, G * N], dim=-1)
    dt = F.softplus(dt + vec(st, m + "dt_bias", dev))                       # [T, H]
    A = -torch.exp(vec(st, m + "A_log", dev))                               # [H]
    D = vec(st, m + "D", dev)
    xs = xs.reshape(T, H, P)
    rep = H // G
    B = B.reshape(T, G, N).repeat_interleave(rep, dim=1)                    # [T, H, N]
    C = C.reshape(T, G, N).repeat_interleave(rep, dim=1)
    S = torch.zeros(H, P, N, device=dev)
    y = torch.empty(T, H, P, device=dev)
    dA = torch.exp(dt * A)                                                  # [T, H]
    for t in range(T):
        S = S * dA[t][:, None, None] + (dt[t][:, None] * xs[t])[:, :, None] * B[t][:, None, :]
        y[t] = (S @ C[t][:, :, None])[..., 0]
    y = y + D[None, :, None] * xs
    y = y.reshape(T, d_in) * F.silu(z)
    yg = y.reshape(T, G, d_in // G)
    yg = yg * torch.rsqrt(yg.pow(2).mean(-1, keepdim=True) + c["layer_norm_epsilon"])
    y = yg.reshape(T, d_in) * vec(st, m + "norm.weight", dev)
    return lin(st, m + "out_proj", y, dev)


def attention(st, L: str, x: torch.Tensor, c: dict, dev) -> torch.Tensor:
    T = x.shape[0]
    Hq, Hkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    m = L + "mixer."
    q = lin(st, m + "q_proj", x, dev).reshape(T, Hq, hd).transpose(0, 1)    # [Hq, T, hd]
    k = lin(st, m + "k_proj", x, dev).reshape(T, Hkv, hd).transpose(0, 1)
    v = lin(st, m + "v_proj", x, dev).reshape(T, Hkv, hd).transpose(0, 1)
    k = k.repeat_interleave(Hq // Hkv, dim=0)
    v = v.repeat_interleave(Hq // Hkv, dim=0)
    s = (q @ k.transpose(1, 2)) * hd ** -0.5
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), 1), float("-inf"))
    o = torch.softmax(s, dim=-1) @ v                                        # [Hq, T, hd]
    return lin(st, m + "o_proj", o.transpose(0, 1).reshape(T, Hq * hd), dev)


def moe(st, L: str, x: torch.Tensor, c: dict, dev, stats: dict | None = None) -> torch.Tensor:
    m = L + "mixer."
    E, K = c["n_routed_experts"], c["num_experts_per_tok"]
    logits = x.float() @ vec(st, m + "gate.weight", dev).T                 # [T, E]
    s = torch.sigmoid(logits)
    choice = s + vec(st, m + "gate.e_score_correction_bias", dev)
    top = torch.topk(choice, K, dim=-1).indices                             # [T, K]
    w = s.gather(1, top)
    if c.get("norm_topk_prob", True):
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    w = w * c["routed_scaling_factor"]
    lat = lin(st, m + "fc1_latent_proj", x, dev)                            # [T, latent]
    acc = torch.zeros_like(lat)
    hit = torch.unique(top).tolist()
    if stats is not None:
        stats["experts"] = stats.get("experts", 0) + len(hit)
    for a in range(0, len(hit), 32):
        grp = hit[a:a + 32]
        ups = many(st, [m + "experts.%d.up_proj" % e for e in grp], dev)
        downs = many(st, [m + "experts.%d.down_proj" % e for e in grp], dev)
        for e, wu, wd in zip(grp, ups, downs):
            rows, slot = torch.where(top == e)
            hh = torch.relu(lat[rows] @ wu.T).square()
            acc.index_add_(0, rows, (hh @ wd.T) * w[rows, slot][:, None])
        del ups, downs
    out = lin(st, m + "fc2_latent_proj", acc, dev)
    sh = torch.relu(lin(st, m + "shared_experts.up_proj", x, dev)).square()
    return out + lin(st, m + "shared_experts.down_proj", sh, dev)


def embed(st, ids: list[int], dev) -> torch.Tensor:
    _, dt, shape, _, _ = st.t[PFX + "embeddings.weight"]
    if dt != "BF16":
        return torch.from_numpy(np.array(st.f32(PFX + "embeddings.weight"))[ids]).to(dev)
    raw = st.raw(PFX + "embeddings.weight").view("<u2").reshape(shape)
    rows = np.array(raw[ids]).astype(np.uint32) << 16
    return torch.from_numpy(rows.view(np.float32).copy()).to(dev)


def head(st, h: torch.Tensor, c: dict, dev) -> torch.Tensor:
    x = rms(h, vec(st, PFX + "norm_f.weight", dev), c["layer_norm_epsilon"])
    return lin(st, "language_model.lm_head", x, dev)


def forward(st, c: dict, ids: list[int], dev, layers: int | None = None, keep: int = 1, want_hs: bool = False,
            verbose: bool = True):
    h = embed(st, ids, dev)
    types = c["layers_block_type"]
    n = len(types) if layers is None else layers
    hs = []
    stats = {}
    for li in range(n):
        t0 = time.time()
        L = "%slayers.%d." % (PFX, li)
        x = rms(h, vec(st, L + "norm.weight", dev), c["layer_norm_epsilon"])
        ty = types[li]
        if ty == "mamba":
            h = h + mamba(st, L, x, c, dev)
        elif ty == "attention":
            h = h + attention(st, L, x, c, dev)
        elif ty == "moe":
            h = h + moe(st, L, x, c, dev, stats)
        else:
            raise SystemExit("layer %d: block type %s" % (li, ty))
        if want_hs:
            hs.append(h[-keep:].cpu().numpy())
        if verbose:
            print("layer %2d %-9s %.1fs  |h| %.3f" % (li, ty, time.time() - t0, h[-1].norm().item()), file=sys.stderr,
                  flush=True)
        if dev != "cpu":
            torch.cuda.empty_cache()
    logits = head(st, h[-keep:], c, dev) if n == len(types) else None
    return logits, hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text")
    ap.add_argument("--ids", help="comma-separated token ids, or @FILE")
    ap.add_argument("--gen", type=int, default=0, help="greedy tokens after the prompt (a whole forward each)")
    ap.add_argument("--layers", type=int, help="run only the first N layers (no logits)")
    ap.add_argument("--dump")
    ap.add_argument("--dump-last", type=int, default=1)
    ap.add_argument("--no-hs", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    md = pathlib.Path(a.model).expanduser()
    cfg = json.loads((md / "config.json").read_text())
    c = cfg.get("llm_config", cfg)
    st = SafeTensors(md)
    tok = None
    if a.ids:
        s = pathlib.Path(a.ids[1:]).read_text() if a.ids.startswith("@") else a.ids
        ids = [int(v) for v in s.replace(",", " ").split()]
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(md / "tokenizer.json"))
        ids = tok.encode(a.text or "The capital of France is", add_special_tokens=False).ids
        if c.get("bos_token_id") is not None:
            ids = [c["bos_token_id"]] + ids
    print("ids (%d): %s" % (len(ids), ",".join(map(str, ids))), file=sys.stderr)
    torch.set_grad_enabled(False)
    keep = min(len(ids), max(1, a.dump_last))
    logits, hs = forward(st, c, ids, a.device, a.layers, keep, a.dump is not None and not a.no_hs)
    if logits is not None:
        top = torch.topk(logits[-1], 10)
        print("top: " + "  ".join("%d:%.3f%s" % (i, v, "" if tok is None else "(%r)" % tok.decode([i]))
                                  for v, i in zip(top.values.tolist(), top.indices.tolist())))
    if a.dump:
        np.savez(a.dump, ids=np.array(ids, dtype=np.int32),
                 logits=logits.cpu().numpy() if logits is not None else np.zeros(0, np.float32),
                 hs=np.stack(hs) if hs else np.zeros(0, np.float32))
    gen = list(ids)
    for _ in range(a.gen if logits is not None else 0):
        nxt = int(torch.argmax(logits[-1]))
        gen.append(nxt)
        print("gen %d%s" % (nxt, "" if tok is None else " %r" % tok.decode([nxt])), flush=True)
        logits, _ = forward(st, c, gen, a.device, None, 1, False, verbose=False)


if __name__ == "__main__":
    main()
