"""Routing calibration for domain pruning (tools/prune_experts.py): run the engine with --dump-routing over a coding /
agentic set - source files teacher-forced through the prompt path, and chat-template coding tasks (some with tool
definitions and tool turns) answered greedily - one routing trace per run in --out.

    python tools/mimo_calib.py --engine build/strata --pack DIR/pack --out /tmp/calib [--code-tokens 120000] [--env K=V]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import jinja2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strata_tokenizer as T  # noqa: E402

LIBS = "/media/per/work/llm/toolchain/ninfer-env/targets/x86_64-linux/lib"
EOS = "151645,151643"
ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True)
ap.add_argument("--pack", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--code-tokens", type=int, default=120000)
ap.add_argument("--chunk", type=int, default=28000)
ap.add_argument("--new", type=int, default=384)
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--root", default=".", help="the repo whose sources are the code part")
ap.add_argument("--code-dirs", default="", help="more source directories (comma separated)")
a = ap.parse_args()

tk = T.Tokenizer.from_dir(a.pack + "/tokenizer")
envj = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
envj.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
tpl = envj.from_string(open(a.pack + "/tokenizer/chat_template.jinja", encoding="utf-8").read())
env = dict(os.environ, LD_LIBRARY_PATH=LIBS + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
env.update(kv.split("=", 1) for kv in a.env)
os.makedirs(a.out, exist_ok=True)


def run(name, ids, new, ctx):
    path = os.path.join(a.out, name + ".ids")
    open(path, "w").write(",".join(map(str, ids)))
    trace = os.path.join(a.out, name + ".routes")
    cmd = [a.engine, "--glm-pack", a.pack, "--tokens-file", path, "--max-new", str(new), "--greedy",
           "--max-context", str(ctx), "--eos-ids", EOS, "--dump-routing", trace]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print("%-10s %6d prompt tokens, %4d new, exit %d, trace %.1f MB" % (
        name, len(ids), sum(1 for l in r.stdout.splitlines() if l.startswith("T ")), r.returncode,
        os.path.getsize(trace) / 1e6 if os.path.exists(trace) else 0), flush=True)
    if r.returncode != 0:
        print(r.stderr[-800:])


# 1. code, teacher-forced in chunks: C / C++ / CUDA / Python not in the KL evaluation (that reads src/core/glm_*.cu and
#    docs/*.md)
files = []
ap2 = a.code_dirs.split(",") if a.code_dirs else []
pats = [os.path.join(a.root, p) for p in ("src/kernels/cuda/*.cu", "src/program/*.cpp", "tools/*.py")]
for d in ap2:   # (other code: e.g. llama.cpp's src/*.cpp and ggml/src/*.c, the Python stdlib)
    pats += [os.path.join(d, "*.c"), os.path.join(d, "*.cpp"), os.path.join(d, "*.py")]
per = []
for pat in pats:
    per.append(sorted(glob.glob(pat)))
# interleave the sources (a file from each in turn) so every language reaches the token budget
while any(per):
    for lst in per:
        if lst:
            files.append(lst.pop(0))
code, n = [], 0
for f in files:
    t = open(f, encoding="utf-8", errors="replace").read()[:12000]   # (no one file dominates)
    code.append("// file: %s\n%s\n" % (os.path.relpath(f, a.root), t))
    n += len(t) // 3
    if n > a.code_tokens * 1.3:
        break
ids = tk.encode("".join(code))[: a.code_tokens]
for i in range(0, len(ids), a.chunk):
    run("code%02d" % (i // a.chunk), ids[i:i + a.chunk], 0, a.chunk + 512)

# 2. coding / agentic chat tasks, answered (the decode's routes too)
tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file of the repository",
                                           "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                          "required": ["path"]}}},
         {"type": "function", "function": {"name": "run_bash", "description": "Run a shell command, return its output",
                                           "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                                                          "required": ["cmd"]}}},
         {"type": "function", "function": {"name": "edit_file", "description": "Replace text in a file",
                                           "parameters": {"type": "object", "properties": {
                                               "path": {"type": "string"}, "old": {"type": "string"},
                                               "new": {"type": "string"}}, "required": ["path", "old", "new"]}}}]
snippet = open(os.path.join(a.root, "tools/strata_tokenizer.py"), encoding="utf-8").read()[:6000]
tasks = [
    "Write a Python function that parses an ISO 8601 duration like P3DT4H5M into seconds, with unit tests.",
    "Implement an LRU cache in C++17 with O(1) get and put, templated on key and value.",
    "Write a bash script that finds the 10 largest files under a directory, skipping .git, and prints sizes.",
    "Explain this error and fix it: TypeError: 'NoneType' object is not subscriptable at line `x = cfg['a']['b']`.",
    "Refactor this JavaScript callback code to async/await: fs.readFile(p, (e, d) => { if (e) cb(e); else cb(null, JSON.parse(d)); });",
    "Write a SQL query: the top 3 customers by total order value per country, from orders(id, customer_id, total) and customers(id, country).",
    "Write a CUDA kernel that computes a row-wise softmax of a float matrix with one warp per row.",
    "Write a Rust function that merges two sorted Vec<i32> into one sorted Vec without allocation beyond the output.",
    "Review this code for bugs:\n\n```python\ndef mean(xs):\n    total = 0\n    for i in range(1, len(xs)):\n        total += xs[i]\n    return total / len(xs)\n```",
    "Write a Go HTTP handler that returns JSON {\"status\": \"ok\", \"time\": <unix seconds>} and a test for it.",
    "Here is a tokenizer module:\n\n```python\n" + snippet + "\n```\n\nSummarize what it does and point out one possible bug.",
    "Write a Makefile for a C project with src/*.c, build/ objects, a debug and a release target.",
    "Write a Python asyncio program that fetches 20 URLs with at most 5 in flight and retries failures twice.",
    "Write a git command sequence to split the last commit into two commits, one per file.",
    "Implement binary search on a rotated sorted array in Java and explain the invariants.",
    "Write a TypeScript type that makes every property of an object type optional and readonly, recursively.",
]
agentic = [
    [{"role": "user", "content": "The test suite fails with ImportError: cannot import name 'Tokenizer' from "
                                 "strata_tokenizer. Find and fix the problem."},
     {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "run_bash",
                                                                      "arguments": {"cmd": "grep -n 'class ' tools/strata_tokenizer.py"}}}]},
     {"role": "tool", "content": "12:class _Bpe:\n180:class Tokeniser:"}],
    [{"role": "user", "content": "Add a --verbose flag to tools/mimo_quality.py that prints each task's full text."},
     {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file",
                                                                      "arguments": {"path": "tools/mimo_quality.py"}}}]},
     {"role": "tool", "content": open(os.path.join(a.root, "tools/mimo_quality.py"), encoding="utf-8").read()[:5000]}],
    [{"role": "user", "content": "The build fails: error: 'uintptr_t' was not declared in this scope in "
                                 "src/kernels/cuda/mimo_attn.cu. Fix it with the smallest change."}],
    [{"role": "user", "content": "Run the unit tests and tell me which ones fail."},
     {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "run_bash",
                                                                      "arguments": {"cmd": "python -m pytest -q tools/"}}}]},
     {"role": "tool", "content": "..F.....F\nFAILED tools/test_calibrate_glm.py::test_usage_env - AssertionError: "
                                 "{'STRATA_GLM_USAGE': '/tmp/x'} != {}\nFAILED tools/test_glm_tier_replay.py::test_lru "
                                 "- KeyError: 3\n2 failed, 7 passed in 1.92s"}],
]
for i, t in enumerate(tasks):
    s = tpl.render(messages=[{"role": "user", "content": t}], add_generation_prompt=True, enable_thinking=False)
    run("task%02d" % i, tk.encode(s, parse_special=True), a.new, 8192)
for i, m in enumerate(agentic):
    s = tpl.render(messages=m, tools=tools, add_generation_prompt=True, enable_thinking=False)
    run("agent%02d" % i, tk.encode(s, parse_special=True), a.new, 8192)
json.dump({"code_tokens": len(ids), "tasks": len(tasks), "agentic": len(agentic)}, open(os.path.join(a.out, "calib.json"), "w"))
