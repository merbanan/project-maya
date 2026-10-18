"""MiMo quality suite (TODO-STATUS Q2-Q4): greedy generations through the chat template (thinking off) - summaries,
a raw-completion summary (the format that looped at 28K), a long story, a 40-item list, code, a math word problem and
three-fact needles - scored for loops (a periodic tail), repeated 4-grams, EOS and a task check.  One engine config a
run (--pack, --env); compare configs by their --out JSON lines.

    python tools/mimo_quality.py --engine build/strata --pack DIR/pack [--env K=V ...] [--tasks a,b] [--out f.jsonl]
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time

import jinja2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import strata_tokenizer as T  # noqa: E402

LIBS = "/media/per/work/llm/toolchain/ninfer-env/targets/x86_64-linux/lib"
EOS = [151645, 151643]   # <|im_end|>, <|endoftext|>

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True)
ap.add_argument("--pack", required=True)
ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the engine")
ap.add_argument("--tasks", default="summ28k,summ8k,raw28k,story,list,code,math,needle8k,needle24k,needle56k")
ap.add_argument("--doc", default="/tmp/mimo_p95000.txt", help="a long text ending in a raw summary instruction")
ap.add_argument("--label", default="")
ap.add_argument("--out", default="")
ap.add_argument("--keep", default="", help="a directory for each task's generated text")
a = ap.parse_args()

tk = T.Tokenizer.from_dir(a.pack + "/tokenizer")
envj = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
envj.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
tpl = envj.from_string(open(a.pack + "/tokenizer/chat_template.jinja", encoding="utf-8").read())


def chat(text):
    s = tpl.render(messages=[{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return tk.encode(s, parse_special=True)


doc = open(a.doc, encoding="utf-8").read()
cut = doc.rfind("\n\n")
body = doc[:cut] if cut > 0 else doc


def head_tokens(text, n):   # the text's first ~n tokens (whole lines)
    out, k = [], 0
    for line in text.splitlines(keepends=True):
        k += len(tk.encode(line))
        if k > n:
            break
        out.append(line)
    return "".join(out)


def needle(n_tok):
    rnd = random.Random(7)
    subj = ["The river", "A merchant", "The old library", "Our committee", "The northern road", "A young engineer",
            "The harbour master", "The orchestra", "A small bakery", "The weather station"]
    verb = ["described", "measured", "ignored", "rebuilt", "celebrated", "documented", "questioned", "repaired"]
    obj = ["the eastern bridge", "a set of ledgers", "the spring festival", "three copper kettles", "the town archive",
           "a broken telescope", "the market square", "an unusual map", "the winter harvest", "a forgotten song"]
    facts = ["The password for the blue vault is MARIGOLD-4471.", "Captain Ilse Varga was born in the city of Tromso.",
             "The experiment's final reading was 93.62 kelvin."]
    chunks, n = [], 0
    while n < n_tok:
        s = "%s %s %s in the year %d. " % (rnd.choice(subj), rnd.choice(verb), rnd.choice(obj), rnd.randint(1200, 1999))
        chunks.append(s)
        n += len(tk.encode(s))
    for depth, f in zip((0.10, 0.50, 0.85), facts):
        chunks.insert(int(len(chunks) * depth), f + " ")
    return ("".join(chunks) + "\n\nUsing only the text above, answer three questions in one line each: 1) the password "
            "for the blue vault, 2) the city where Captain Ilse Varga was born, 3) the experiment's final reading.")


def check_list(t):
    items = [re.sub(r"^\s*\d+[.)]\s*", "", l).strip().lower() for l in t.splitlines() if re.match(r"^\s*\d+[.)]", l)]
    return {"items": len(items), "distinct": len(set(items)), "ok": len(set(items)) >= 40}


def check_code(t):
    m = re.search(r"```(?:python)?\n(.*?)```", t, re.S)
    src = m.group(1) if m else t
    ns = {}
    try:
        exec(compile(src, "<gen>", "exec"), ns)   # (the model's own function, run on fixed inputs)
        f = ns["merge_intervals"]
        ok = f([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]] and f([[1, 4], [4, 5]]) == [[1, 5]]
        return {"ok": bool(ok)}
    except Exception as e:   # noqa: BLE001
        return {"ok": False, "err": type(e).__name__}


TASKS = {
    "summ28k": (lambda: chat(body + "\n\nSummarize the text above in three sentences."), 512, None, 32768),
    "summ8k": (lambda: chat(head_tokens(body, 8000) + "\n\nSummarize the text above in three sentences."), 512, None,
               16384),
    "raw28k": (lambda: tk.encode(doc), 512, None, 32768),
    "story": (lambda: chat("Write a short story of about 700 words about a lighthouse keeper who finds a radio that "
                           "receives tomorrow's weather."), 1024, None, 4096),
    "list": (lambda: chat("List 40 different animals that live in the ocean, one per line, numbered 1 to 40. No "
                          "other text."), 512, check_list, 4096),
    "code": (lambda: chat("Write a Python function merge_intervals(intervals) that takes a list of [start, end] "
                          "pairs and returns the merged overlapping intervals sorted by start (touching intervals "
                          "merge). Output only the code in one ```python block."), 512, check_code, 4096),
    "math": (lambda: chat("A train leaves at 09:40 and travels 237 km at a constant 79 km/h, then waits 25 minutes, "
                          "then travels 96 km at 64 km/h. At what time does it arrive? Show the steps briefly and give "
                          "the final answer as HH:MM on the last line."), 512,
             lambda t: {"ok": "14:35" in t.strip().splitlines()[-1] if t.strip() else False}, 4096),
    "needle8k": (lambda: chat(needle(8000)), 96, None, 16384),
    "needle24k": (lambda: chat(needle(24000)), 96, None, 32768),
    "needle56k": (lambda: chat(needle(56000)), 96, None, 65536),
}


def needle_check(t):
    hit = [k in t for k in ("MARIGOLD-4471", "Tromso", "93.62")]
    return {"hits": sum(hit), "ok": all(hit)}


def loop_of(g):
    """the tail's longest periodic run: (period, run length) over periods 1..256 - a loop when run >= max(2p, 32)"""
    best = (0, 0)
    for p in range(1, min(256, len(g) // 2) + 1):
        k = 0
        while k + p < len(g) and g[len(g) - 1 - k] == g[len(g) - 1 - k - p]:
            k += 1
        if k >= max(2 * p, 32) and k > best[1]:
            best = (p, k)
    return best


def rep4(g):
    grams = [tuple(g[i:i + 4]) for i in range(len(g) - 3)]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


env = dict(os.environ, LD_LIBRARY_PATH=LIBS + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
env.update(kv.split("=", 1) for kv in a.env)
for name in a.tasks.split(","):
    build, new, check, ctx = TASKS[name]
    if name.startswith("needle"):
        check = needle_check
    ids = build()
    path = "/tmp/mimoq_%d.ids" % os.getpid()
    open(path, "w").write(",".join(map(str, ids)))
    t0 = time.time()
    r = subprocess.run([a.engine, "--glm-pack", a.pack, "--tokens-file", path, "--max-new", str(new), "--greedy",
                        "--max-context", str(ctx), "--eos-ids", ",".join(map(str, EOS))],
                       capture_output=True, text=True, env=env)
    gen = [int(m.group(1)) for m in re.finditer(r"^T (\d+)", r.stdout, re.M)]
    eos = bool(gen) and (gen[-1] in EOS or len(gen) < new)   # (stopped early: an EOS, printed or not)
    body_ids = [t for t in gen if t not in EOS]
    text = tk.decode(body_ids)
    lp, lk = loop_of(body_ids)
    res = {"label": a.label, "task": name, "prompt": len(ids), "gen": len(gen), "eos": eos, "loop_p": lp,
           "loop_run": lk, "rep4": round(rep4(body_ids), 3), "s": round(time.time() - t0, 1), "exit": r.returncode}
    if check:
        res.update(check(text))
    print("%-10s %-10s prompt %6d gen %5d eos %-5s loop %-9s rep4 %.3f %s" % (
        a.label, name, len(ids), len(gen), eos, ("p%d x%d" % (lp, lk)) if lp else "-", res["rep4"],
        " ".join("%s=%s" % (k, res[k]) for k in ("ok", "hits", "distinct", "err") if k in res)), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
    if a.keep:
        os.makedirs(a.keep, exist_ok=True)
        open(os.path.join(a.keep, "%s_%s.txt" % (a.label or "run", name)), "w").write(text)
    if r.returncode != 0:
        print("  engine stderr tail:", r.stderr[-600:].replace("\n", " | "))
