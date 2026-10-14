"""DSv4 verify-cost probe: how long n+1 tokens take in one pass (the prompt path from a snapshot, as an MTP verify of
n drafts would) against n+1 decode steps.  python d4verify.py [TOKENS]"""
import os, subprocess, sys, time
sys.path.insert(0, "tools")
import strata_tokenizer as T
D = "/media/per/work/llm/Maya-data/models/DSv4-Flash"
tk = T.Tokenizer.from_dir(D + "/pack/tokenizer")
n_tok = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
text = ("The committee met to discuss the river, the market and the old library. " * 400)
ids = tk.encode("<｜begin▁of▁sentence｜><｜User｜>" + text + "\nSummarize the text above in detail.<｜Assistant｜></think>",
                parse_special=True)[:n_tok]
env = dict(os.environ, LD_LIBRARY_PATH="/media/per/work/llm/toolchain/ninfer-env/targets/x86_64-linux/lib",
           CUDA_MODULE_LOADING="LAZY")
p = subprocess.Popen(["build/strata", "--glm-pack", D + "/pack", "--serve", "--max-context", "16384", "--eos-ids", "1"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/tmp/d4verify.err", "w"), text=True, env=env)
for line in p.stdout:
    if line.startswith(("READY", "ERR")): break
def gen(prompt, n):
    p.stdin.write("GEN %d temperature=0 %s\n" % (n, ",".join(map(str, prompt)))); p.stdin.flush()
    toks = []
    for line in p.stdout:
        if line.startswith("T "): toks.append(int(line.split()[1]))
        elif line.startswith(("DONE", "ERR")):
            f = line.split()
            return toks, float(f[3]), float(f[4]), int(f[8]) if len(f) > 8 else -1
gen(ids, 8)                                   # warm
cont, pms, dms, _ = gen(ids, 72)
dec = dms / max(1, len(cont) - 1)
print("decode: %.2f ms a token (%d tokens)" % (dec, len(cont)))
for n in (1, 2, 4, 8):
    gen(ids, 1)                               # the snapshot back at the prompt
    _, pms, _, reuse = gen(ids + cont[:n], 1)
    print("verify of %d drafts (%d tokens in one pass): %.1f ms = %.2f decode steps (reuse %d)" % (n, n + 1, pms, pms / dec, reuse))
p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=60)
