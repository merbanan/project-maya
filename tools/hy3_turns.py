"""Agent-turn prompt time on Hy3 (see glm_turns.py): a 12K prompt and its answer, then history + a short user turn."""
import os, subprocess, sys
import jinja2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strata_tokenizer as T
D = os.environ.get("HY3_D", "/media/per/work/llm/Maya-data/models/Hy3-bartowski-IQ2_XXS")
tk = T.Tokenizer.from_dir(D + "/pack/tokenizer")
envj = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
envj.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
tpl = envj.from_string(open(D + "/pack/tokenizer/chat_template.jinja", encoding="utf-8").read())
def turn(q, first):
    t = tpl.render(messages=[{"role": "user", "content": q}], add_generation_prompt=True)
    ids = tk.encode(t, parse_special=True)
    return ids if first else ids[1:]   # (later turns without the leading BOS)
env = dict(os.environ, LD_LIBRARY_PATH="/media/per/work/llm/toolchain/ninfer-env/targets/x86_64-linux/lib",
           STRATA_GLM_RAM_HEADROOM_GB="4", STRATA_GLM_RESERVE_MB="400", STRATA_GLM_KV="tq6v3", CUDA_MODULE_LOADING="LAZY")
p = subprocess.Popen(["build/strata", "--glm-pack", D + "/pack", "--serve", "--max-context", "16384"], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=open("/tmp/hy3turn.err", "w"), text=True, env=env)
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
h = turn(FILL + "\nSummarize in one line.", True)
a, _ = gen(h, 32)
for q in ["Now list three key words.", "Translate the summary to French.", "Write it as a haiku."]:
    h = h + a + turn(q, False)
    a, f = gen(h, 32)
    print("turn: %s" % " ".join(f[:8]), flush=True)
p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=120)
