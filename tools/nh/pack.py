"""tools/nh/pack.py - maya-nh's pack for an NVIDIA-Nemotron-3.5 (Nemotron-H) EXL3 model, beside the model's files (no
weights copied).

    python tools/nh/pack.py --model ~/models/Nemotron-3.5-Super-VL-EXL3-4.23bpw

writes <model>/maya-nh/:
  nh.txt        the geometry, `key value` lines (lists comma-separated), from config.json's llm_config (the
                block types as layers_block_type, one letter each: M mamba, A attention, E moe), the special token
                ids the engine needs and the experts' EXL3 bits (exl3.experts <up>,<down>: every MoE layer alike)
  tensors.txt   every tensor of the model's safetensors: `file <i> <path>` lines (paths relative to the model
                directory), then `t <name> <dtype> <file> <offset> <bytes> <dims...>` (src/mm3/mm3_pack.hpp's format)
  tokenizer/    the server's tokenizer (tools/strata_tokenizer.py's layout, pre "nemotron") and the model's chat
                template as it ships

The pack stops on anything the engine does not cover: an EXL3 matrix whose codebook is not mul1 or whose bits are
not whole, experts at different bits, a block type other than mamba / attention / moe, a layer without its tensors.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mm3"))
from st import SafeTensors  # noqa: E402

MUL1 = 0x83DCD12D
P = "language_model.backbone."
LETTER = {"mamba": "M", "attention": "A", "moe": "E"}


def need(st: SafeTensors, c: dict) -> list[str]:
    """The tensors the engine reads; a missing one stops the pack here instead of the engine's start."""
    miss = []

    def want(name: str):
        if not st.has(name):
            miss.append(name)

    for li, ty in enumerate(c["layers_block_type"]):
        L = "%slayers.%d." % (P, li)
        m = L + "mixer."
        want(L + "norm.weight")
        if ty == "mamba":
            for n in ("in_proj", "out_proj"):
                want(m + n + ".trellis")
            for n in ("conv1d.weight", "conv1d.bias", "A_log", "D", "dt_bias", "norm.weight"):
                want(m + n)
        elif ty == "attention":
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                want(m + n + ".trellis")
        elif ty == "moe":
            for n in ("fc1_latent_proj", "fc2_latent_proj", "shared_experts.up_proj", "shared_experts.down_proj"):
                want(m + n + ".trellis")
            want(m + "gate.weight")
            want(m + "gate.e_score_correction_bias")
            for e in range(c["n_routed_experts"]):
                for n in ("up_proj", "down_proj"):
                    want(m + "experts.%d.%s.trellis" % (e, n))
    want(P + "embeddings.weight")
    want(P + "norm_f.weight")
    want("language_model.lm_head.trellis")
    return miss


def check_exl3(st: SafeTensors) -> tuple[int, list[str]]:
    """Every EXL3 matrix: mul1, whole bits, suh / svh beside it. -> (matrices, problems)."""
    bad, n = [], 0
    for name, (_, _, shape, _, _) in st.t.items():
        if not name.endswith(".trellis"):
            continue
        n += 1
        pre = name[:-len(".trellis")]
        if not st.has(pre + ".mul1"):
            bad.append("%s: codebook %s, the engine decodes mul1" % (pre, "mcg" if st.has(pre + ".mcg") else "3inst"))
        elif int(st.raw(pre + ".mul1").view("<u4")[0]) != MUL1:
            bad.append("%s: mul1 multiplier 0x%X" % (pre, int(st.raw(pre + ".mul1").view("<u4")[0])))
        if shape[-1] % 16:
            bad.append("%s: %g bits (fractional trellis)" % (pre, shape[-1] / 16))
        for s in (".suh", ".svh"):
            if not st.has(pre + s):
                bad.append(pre + s + " missing")
    return n, bad


def expert_bits(st: SafeTensors, c: dict) -> tuple[tuple[int, int], list[str]]:
    """(up K, down K) of the routed experts - one pair for every MoE layer (the engine's experts share one layout)."""
    ks = {"up_proj": set(), "down_proj": set()}
    for li, ty in enumerate(c["layers_block_type"]):
        if ty != "moe":
            continue
        m = "%slayers.%d.mixer.experts." % (P, li)
        for w in ks:
            ks[w] |= {st.t["%s%d.%s.trellis" % (m, e, w)][2][-1] // 16 for e in range(c["n_routed_experts"])}
    bad = ["experts' %s at bits %s" % (w, sorted(v)) for w, v in ks.items() if len(v) != 1]
    return (max(ks["up_proj"]), max(ks["down_proj"])), bad


def write_tokenizer(out: pathlib.Path, model: pathlib.Path) -> dict:
    """vocab.json (id order), merges.txt, token_type.json, tokenizer.json {pre, special_ids} and the chat template."""
    tj = json.loads((model / "tokenizer.json").read_text(encoding="utf-8"))
    pre = tj.get("pre_tokenizer") or {}
    parts = pre.get("pretokenizers", [])
    split = [p for p in parts if p.get("type") == "Split"]
    if (tj.get("normalizer") is not None or len(split) != 1 or split[0].get("behavior") != "Isolated"
            or tj["model"].get("type") != "BPE"):
        raise SystemExit("tokenizer.json: not the plain BPE + one Isolated Split the `nemotron` scheme transcribes")
    sys.path.insert(0, str(HERE.parent))
    from strata_tokenizer import NEMOTRON_PATTERN
    if split[0]["pattern"].get("Regex") != NEMOTRON_PATTERN:
        raise SystemExit("tokenizer.json: its Split regex is not the one strata_tokenizer.py's `nemotron` transcribes")
    vocab = tj["model"]["vocab"]
    added = {a["id"]: a for a in tj["added_tokens"]}
    n = max(max(vocab.values()), max(added)) + 1
    tokens = ["<|unused_%d|>" % i for i in range(n)]
    types = [1] * n                                            # llama.cpp: 1 normal, 3 control, 4 user-defined
    for s, i in vocab.items():
        tokens[i] = s
    for i, a in added.items():
        tokens[i] = a["content"]
        types[i] = 3 if a["special"] else 4
    merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
    if len(set(tokens)) != n:
        raise SystemExit("tokenizer.json: duplicate token strings")
    tc = json.loads((model / "tokenizer_config.json").read_text())
    ids = {a["content"]: i for i, a in added.items()}
    special = {k[:-len("_token")]: ids[v if isinstance(v, str) else v["content"]]
               for k, v in tc.items() if k.endswith("_token") and v is not None}
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}, ensure_ascii=False),
                                    encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(types))
    (out / "tokenizer.json").write_text(json.dumps({"pre": "nemotron", "special_ids": special}, indent=1))
    (out / "chat_template.jinja").write_text((model / "chat_template.jinja").read_text(encoding="utf-8"),
                                             encoding="utf-8")
    return special


def flat(f, prefix: str, d: dict):
    for k, v in d.items():
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, (int, float)):
            f.write("%s%s %s\n" % (prefix, k, v))
        elif isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            f.write("%s%s %s\n" % (prefix, k, ",".join(str(int(x) if isinstance(x, bool) else x) for x in v)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", help="default: <model>/maya-nh")
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser() if a.out else model / "maya-nh"
    cfg = json.loads((model / "config.json").read_text())
    c = cfg.get("llm_config", cfg)
    q = cfg.get("quantization_config", {})
    if q.get("quant_method") != "exl3":
        raise SystemExit("%s: not an EXL3 model (quant_method %r)" % (model, q.get("quant_method")))
    if c.get("model_type") != "nemotron_h":
        raise SystemExit("%s: model_type %r, not nemotron_h" % (model, c.get("model_type")))
    types = c["layers_block_type"]
    if any(t not in LETTER for t in types):
        raise SystemExit("block types %s: the engine runs mamba, attention and moe" % sorted(set(types) - set(LETTER)))
    if c.get("n_group", 1) != 1 or c.get("topk_group", 1) != 1:
        raise SystemExit("grouped routing (n_group %r) is not covered" % c.get("n_group"))
    if c.get("mlp_hidden_act") != "relu2" or c.get("mamba_hidden_act") != "silu":
        raise SystemExit("activations %r / %r: the engine runs relu2 / silu" % (c.get("mlp_hidden_act"),
                                                                                c.get("mamba_hidden_act")))
    st = SafeTensors(model)
    miss = need(st, c)
    if miss:
        raise SystemExit("missing tensors (%d):\n  %s" % (len(miss), "\n  ".join(miss[:20])))
    nmat, bad = check_exl3(st)
    bits, bad2 = expert_bits(st, c)
    if bad or bad2:
        raise SystemExit("not covered:\n  " + "\n  ".join((bad + bad2)[:20]))
    out.mkdir(parents=True, exist_ok=True)
    special = write_tokenizer(out / "tokenizer", model)
    gen = json.loads((model / "generation_config.json").read_text())
    eos = gen["eos_token_id"] if isinstance(gen["eos_token_id"], list) else [gen["eos_token_id"]]
    with (out / "nh.txt").open("w") as f:
        flat(f, "", {k: v for k, v in c.items() if k != "layers_block_type"})
        f.write("layers_block_type %s\n" % "".join(LETTER[t] for t in types))
        f.write("num_hidden_layers %d\n" % len(types))
        f.write("bos_token_id %d\n" % gen["bos_token_id"])
        f.write("eos_token_id %s\n" % ",".join(map(str, eos)))
        for k in ("img_context_token_id",):
            if isinstance(cfg.get(k), int):
                f.write("%s %d\n" % (k, cfg[k]))
        for k, v in special.items():
            f.write("special.%s %d\n" % (k, v))
        f.write("exl3.experts %d,%d\n" % bits)
        f.write("mtp %d\n" % int(st.has("language_model.mtp.layers.0.eh_proj.trellis")))
        f.write("model_dir %s\n" % model)
    with (out / "tensors.txt").open("w") as f:
        fidx = {}
        for i, fn in enumerate(st.files):
            fidx[fn] = i
            f.write("file %d %s\n" % (i, fn))
        for name, (fn, dt, shape, off, nb) in st.t.items():
            f.write("t %s %s %d %d %d %s\n" % (name, dt, fidx[fn], off, nb, " ".join(map(str, shape))))
    print("%s: %d tensors in %d files (%d EXL3 matrices, all mul1, whole bits) -> %s" %
          (model.name, len(st.t), len(st.files), nmat, out))
    print("  blocks: %s" % "".join(LETTER[t] for t in types))
    print("  experts' bits (up, down): %d/%d" % bits)
    print("  special ids: %s" % special)
    return 0


if __name__ == "__main__":
    sys.exit(main())
