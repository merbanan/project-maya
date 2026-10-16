"""Agent-turn prompt time: a 12K prompt and its answer, then the same session gets that history + a short user turn
(the engine keeps the prefix); prints each later turn's prompt ms.  Env as glmspec.py."""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strata_tokenizer as T
D = "/media/per/work/llm/Maya-data/models/Maya-S24"
tk = T.Tokenizer.from_dir(D + "/pack/tokenizer")
ENV = {"STRATA_GLM_RAM_HEADROOM_GB": "4", "STRATA_GLM_PREFILL_SUB": "1024", "STRATA_GLM_KV_INT8": "1",
       "STRATA_GLM_RESERVE_MB": "400", "STRATA_GLM_CPU_PLAN": "012334567"}
env = dict(os.environ, LD_LIBRARY_PATH="/media/per/work/llm/toolchain/ninfer-env/targets/x86_64-linux/lib", **ENV)
p = subprocess.Popen(["build/strata", "--glm-pack", D + "/pack", "--serve", "--max-context", "16384"], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=open("/tmp/glmturn.err", "w"), text=True, env=env)
for line in p.stdout:
    if line.startswith(("READY", "ERR")): break
def gen(ids, n):
    p.stdin.write("GEN %d temperature=0 %s\n" % (n, ",".join(map(str, ids)))); p.stdin.flush()
    toks = []
    for line in p.stdout:
        if line.startswith("T "): toks.append(int(line.split()[1]))
        elif line.startswith(("DONE", "ERR")):
            return toks, line.split()
FILL = "The committee met to discuss the river, the market and the old library. " * 900
h = tk.encode("[gMASK]<sop><|user|>" + FILL + "\nSummarize in one line.<|assistant|></think>", parse_special=True)
a, _ = gen(h, 32)
for q in ["Now list three key words.", "Translate the summary to French.", "Write it as a haiku."]:
    h = h + a + tk.encode("<|user|>" + q + "<|assistant|></think>", parse_special=True)
    a, f = gen(h, 32)
    print("turn: %s" % " ".join(f[:8]), flush=True)
p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=120)
