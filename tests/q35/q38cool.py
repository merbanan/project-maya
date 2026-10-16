"""Decode speed at a long context without the prompt's heat: the prompt read with one token generated, the GPU left to
cool below COOL_C, then the same history continued (the engine keeps it) for N tokens.  q38cool.py TOKENS N [env K=V..]"""
import os, subprocess, sys, time
sys.path.insert(0, "tools")
from strata_tokenizer import Tokenizer
G = os.environ.get("Q38_GGUF", "/media/per/work/llm/lucebox-hub/server/models/Qwen3.8-27B-Q4_K_M.gguf")
n_tok, n_new = int(sys.argv[1]), int(sys.argv[2])
env = dict(os.environ); env.update(kv.split("=", 1) for kv in sys.argv[3:])
tk = Tokenizer.from_gguf(G)
text = "<|im_start|>user\n" + "The committee met to discuss the river, the market and the old library. " * (n_tok // 15) + "\nSummarize.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = tk.encode(text, parse_special=True)
extra = [a for a in os.environ.get("Q38_ARGS", "").split() if a]
p = subprocess.Popen(["build/strata-dense", "--serve", "--native", G, "--context", str(len(ids) + n_new + 512)] + extra,
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/tmp/q38cool.err", "w"), text=True, env=env)
for line in p.stdout:
    if line.startswith(("READY", "ERR")): break
def gen(x, n):
    p.stdin.write("GEN %d temperature=0 ignore_eos=1 %s\n" % (n, ",".join(map(str, x)))); p.stdin.flush()
    toks = []
    for line in p.stdout:
        if line.startswith("T "): toks.append(int(line.split()[1]))
        elif line.startswith(("DONE", "ERR")): return toks, line.split()
t, f = gen(ids, 1)
print("prompt %d tokens in %s ms" % (len(ids), f[3]), flush=True)
lim = int(os.environ.get("COOL_C", "60"))
for _ in range(150):
    c = int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip())
    if c <= lim: break
    time.sleep(2)
t2, f2 = gen(ids + t, n_new)
print("decode at %d: %d tokens in %s ms = %.1f tok/s (resumed %s)" % (len(ids), len(t2), f2[4], 1000 * (len(t2) - 1) / float(f2[4]), f2[7] if len(f2) > 7 else "?"), flush=True)
for l in open("/tmp/q38cool.err"):
    if "rounds" in l: print(" ", l.strip()[:220])
p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=60)
