"""DSv4 MTP shadow probe: the same greedy generations without and with STRATA_DSV4_MTP (the block drafts the next
token beside every decode step and counts how often the trunk agrees); the tokens must not change.
D4_SPEC=1: the speculative decode on it instead (STRATA_DSV4_MTP_SPEC); D4_FIX=1: fixed tiers (an exact A/B).
python d4mtp.py [NEW]"""
import os, subprocess, sys
sys.path.insert(0, "tools")
import strata_tokenizer as T
D = "/media/per/work/llm/Maya-data/models/DSv4-Flash"
MTP = D + "/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
tk = T.Tokenizer.from_dir(D + "/pack/tokenizer")
n_new = int(sys.argv[1]) if len(sys.argv) > 1 else 256
PROMPTS = ["Write a Python function that parses a CSV file and returns the average of every numeric column. Explain it.",
           "Explain how a refrigerator works, step by step, for a curious ten year old.",
           "List the first twenty prime numbers and describe the sieve of Eratosthenes."]
FIX = {"STRATA_GLM_RAM_GB": "50", "STRATA_GLM_CPU_LANE": "0"} if os.environ.get("D4_FIX") else {}
LONG = int(os.environ.get("D4_LONG", "0"))   # D4_LONG=<tokens>: each prompt behind that much filler text
FILL = ("The committee met to discuss the river, the market and the old library. " * 4000)
def run(extra):
    env = dict(os.environ, LD_LIBRARY_PATH="/media/per/work/llm/toolchain/ninfer-env/targets/x86_64-linux/lib",
               CUDA_MODULE_LOADING="LAZY")
    env.update(FIX); env.update(extra)
    err = "/tmp/d4mtp%s.err" % ("1" if "STRATA_DSV4_MTP" in extra else "0")
    p = subprocess.Popen(["build/strata", "--glm-pack", D + "/pack", "--serve", "--max-context", os.environ.get("D4_CTX", "16384"), "--eos-ids", "1"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(err, "w"), text=True, env=env)
    for line in p.stdout:
        if line.startswith(("READY", "ERR")):
            if line.startswith("ERR"): print(line.strip())
            break
    out = []
    for q in PROMPTS:
        ids = tk.encode("<｜begin▁of▁sentence｜><｜User｜>" + q + "<｜Assistant｜></think>", parse_special=True)
        if LONG:
            fill = tk.encode(FILL)[:LONG]
            ids = ids[:3] + fill + ids[3:]
        p.stdin.write("GEN %d temperature=0 ignore_eos=1 %s\n" % (n_new, ",".join(map(str, ids)))); p.stdin.flush()
        toks = []
        for line in p.stdout:
            if line.startswith("T "): toks.append(int(line.split()[1]))
            elif line.startswith(("DONE", "ERR")):
                f = line.split()
                if line.startswith("ERR"): print(line.strip())
                dms = float(f[4]) if len(f) > 4 else 0
                print("  %s: %d tokens, decode %.1f tok/s" % ("mtp" if extra else "plain", len(toks), (len(toks) - 1) * 1000 / max(dms, 1e-9)))
                break
        out.append(toks)
    p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=60)
    for l in open(err):
        if "dsv4 mtp" in l: last = l.rstrip(); print("  " + last) if "VRAM" in l else None
    lines = [l.rstrip() for l in open(err) if "accepted" in l or "error" in l.lower()]
    if lines: print("  " + lines[-1])
    return out
a = run({})
b = run({"STRATA_DSV4_MTP": MTP, "STRATA_DSV4_MTP_SPEC": "1" if os.environ.get("D4_SPEC") else "0"})
for i, (x, y) in enumerate(zip(a, b)):
    n = next((k for k, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
    print("prompt %d: %d / %d leading tokens agree" % (i, n, min(len(x), len(y))))
print("tokens unchanged:", a == b)
