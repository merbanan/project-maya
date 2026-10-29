"""serve/test_loop_guard.py - the loop guard (serve/loop_guard.py): what it calls a loop and what not, and what the
server does with one - against engines that continue an answer from wherever the prompt says it got to.

    python -m unittest serve.test_loop_guard -v
"""
from __future__ import annotations

import itertools
import json
import os
import random
import re
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import loop_guard as LG  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (IM_END, ByteTokenizer, Service, check_request_sampling, loop_guard_setting,  # noqa: E402
                          serve)

ROOT = Path(__file__).resolve().parents[1]


class Words:
    """A stand-in for a BPE vocabulary: a word with its leading space, a run of punctuation or of white space."""
    PIECE = re.compile(r" ?\w+| ?[^\w\s]+|\s+")

    def __init__(self):
        self.ids: dict[str, int] = {}

    def __call__(self, text: str) -> list[int]:
        return [self.ids.setdefault(p, len(self.ids)) for p in self.PIECE.findall(text)]


def first_loop(ids):
    """(index of the token at which the detector fires, Loop) or None."""
    det = LG.LoopDetector()
    for i, t in enumerate(ids):
        loop = det.push(t)
        if loop is not None:
            return i, loop
    return None


def numbers_table(rows):
    return "| id | name | score | city |\n|---|---|---|---|\n" + "".join(
        f"| {i} | user{i * 7 % 13} | {(i * 37) % 100} | city{i % 5} |\n" for i in range(rows))


LONG_BLOCK = " ".join(random.Random(11).choices(   # a plan of several hundred tokens (Maya-S thought one 4 times)
    "check the metric logs then the deploy diff and the gc pauses before the p99 of each host".split(), k=600)) + "\n"

LEGIT = {
    "markdown table": numbers_table(60),
    "a long block twice": "Draft:\n" + LONG_BLOCK + "Final:\n" + LONG_BLOCK + "Done.",
    "JSON array": json.dumps(list(range(300))),
    "JSON objects": json.dumps([{"id": i, "name": f"item {i}", "tags": ["a", "b"], "price": i * 3}
                                for i in range(40)], indent=2),
    "one function name": "".join(f"result = process_item(items[{i}], options)\n" for i in range(80)),
    "similar code lines": "".join(f"    x[{i}] = a[{i}] * w[{i % 4}] + b[{i}];\n" for i in range(64)),
    "a block twice": ("def area(r):\n    \"\"\"The area of a circle of radius r.\"\"\"\n    return 3.14159 * r * r\n\n"
                      * 2) + "print(area(2))\n",
    "a short line three times": "Hello there, world!\n" * 3 + "Done.",
    "a short row of zeros": "board = [" + ", ".join(["0"] * 30) + "]\n",
    "prose": " ".join(random.Random(7).choices(
        "the a model token answer loop guard server engine test of to and in on with for is are was".split(), k=3000)),
}


class Detection(unittest.TestCase):
    def test_real_loops_fire(self):
        w = Words()
        cases = {   # text before the loop, the block, at least how many copies there are
            "number sequence": ("The values are: ", "1017, 245, 26, 100, 47, 187, ", 40),
            "sentence in thinking": ("Let me think. ", "Wait, I should double-check the second step again. ", 20),
            "paragraph": ("Summary:\n", "- Also: logging added to a path that previously short-circuited (e.g., "
                                        "cached response now logs and populates cache differently)\n", 10),
            "code block": ("```python\n", "def step(state):\n    state.count += 1\n    if state.count > 10:\n"
                                          "        state.reset()\n    return state\n\n", 10),
            "one token": ("Result: ", "0 ", 300),
            "a long plan in thinking": ("Let me plan. ", LONG_BLOCK, 3),
        }
        for name, (head, block, n) in cases.items():
            with self.subTest(name):
                h, b = w(head), w(block)
                ids = h + b * n
                got = first_loop(ids)
                self.assertIsNotNone(got, name)
                at, loop = got
                p = len(b)
                period = p if p >= LG.P_MIN else p * -(-LG.P_MIN // p)    # (a short one as its multiple)
                self.assertEqual(loop.period, period)
                reps = LG.REPS_SHORT if period < LG.SHORT_BELOW else LG.REPS_LONG
                copies = max(reps, -(-LG.MIN_SPAN // period))
                # it fires on the last token of the copies it needs, and not before
                self.assertLessEqual(at + 1 - len(h), copies * period + period)
                self.assertGreaterEqual(loop.copies, reps)
                self.assertGreaterEqual(loop.repeats + loop.period, LG.MIN_SPAN)

    def test_legit_repetition_does_not_fire(self):
        for name, text in LEGIT.items():
            with self.subTest(name):
                self.assertIsNone(first_loop(Words()(text)), name)
        # byte by byte too (the tests' tokenizer below), the harder case: a byte block is longer in tokens
        for name, text in LEGIT.items():
            if name != "prose":
                with self.subTest(name + " (bytes)"):
                    self.assertIsNone(first_loop(list(text.encode())), name)

    def test_hold_covers_the_repeats(self):
        w = Words()
        head, block = w("Here we go:\n"), w("alpha beta gamma delta epsilon zeta eta theta iota kappa ")
        det, held = LG.LoopDetector(), []
        for t in head + block * 12:
            loop = det.push(t)
            if loop is not None:
                # everything that is dropped was still held back a token earlier
                self.assertGreaterEqual(det.hold, loop.repeats)
                break
            held.append(det.hold)
        self.assertIsNotNone(loop)
        self.assertEqual(held[:len(head) + len(block)], [0] * (len(head) + len(block)))   # nothing repeats yet
        prose = Words()(LEGIT["prose"])
        det = LG.LoopDetector()
        holds = [det.push(t) or det.hold for t in prose]
        # ordinary text: hardly held at all (this prose draws on 22 words, so a 2048-token window finds a few more
        # chance runs of a token or two than 400 did: 2.01 against 1.9; recorded answers hold back the same)
        self.assertLess(sum(holds) / len(holds), 2.5)

    def test_reset_keeps_checking(self):
        w = Words()
        block = w("one two three four five six seven eight nine ten eleven twelve ")
        det = LG.LoopDetector()
        ids = w("start\n") + block * 9
        loops = [det.push(t) for t in ids]
        self.assertIsNotNone(loops[-1])
        kept = ids[:len(ids) - loops[-1].repeats]
        det.reset(kept)
        self.assertEqual(det.hold, 0)
        self.assertTrue(any(det.push(t) for t in block * 9))       # the same loop again is seen again


class Policy(unittest.TestCase):
    def loop_ids(self, n=12):
        return list(range(100, 100 + 20)) * n

    def fire(self, g, ids):
        for t in ids:
            act = g.push(t)
            if act is not None:
                return act
        return None

    def test_answer_penalty_then_stop(self):
        g = LG.LoopGuard()
        a1 = self.fire(g, self.loop_ids())
        self.assertEqual((a1.kind, a1.where, a1.period), (LG.PENALTY, "answer", 20))
        g.resume(list(range(100, 120)), a1)
        a2 = self.fire(g, self.loop_ids())
        self.assertEqual(a2.kind, LG.STOP)

    def test_thinking_penalty_close_then_stop(self):
        g = LG.LoopGuard(think_end=7, in_think=True)
        a1 = self.fire(g, self.loop_ids())
        self.assertEqual((a1.kind, a1.where), (LG.PENALTY, "thinking"))
        g.resume([], a1)
        a2 = self.fire(g, self.loop_ids())
        self.assertEqual((a2.kind, a2.where), (LG.CLOSE_THINKING, "thinking"))
        g.resume([7], a2)
        self.assertFalse(g.in_think)
        a3 = self.fire(g, self.loop_ids())
        self.assertEqual((a3.kind, a3.where), (LG.STOP, "answer"))

    def test_a_loop_in_the_answer_after_one_in_the_thinking(self):
        g = LG.LoopGuard(think_end=7, in_think=True)
        g.resume([], self.fire(g, self.loop_ids()))
        self.assertIsNone(g.push(7))                                # the model closes its thinking itself
        a2 = self.fire(g, self.loop_ids())
        self.assertEqual((a2.kind, a2.where), (LG.PENALTY, "answer"))   # the answer's first: a penalty
        g.resume([], a2)
        self.assertEqual(self.fire(g, self.loop_ids()).kind, LG.STOP)   # the third in all

    def test_no_thinking_token_no_thinking_close(self):
        g = LG.LoopGuard(think_end=None, in_think=True)            # (a model without one: the answer's rules)
        g.resume([], self.fire(g, self.loop_ids()))
        self.assertEqual(self.fire(g, self.loop_ids()).kind, LG.STOP)

    def test_sampling(self):
        g = LG.LoopGuard()
        base = {"temperature": 0, "top_k": 20}
        self.assertIs(g.sampling(base, 0, 500)[0], base)            # before a loop: the request's own, as it is
        self.assertEqual(g.sampling(base, 0, 500)[1], 500)
        act = self.fire(g, self.loop_ids())
        kept = list(range(1000))
        g.resume(kept, act)
        smp, n = g.sampling(base, len(kept), 5000)
        self.assertEqual(smp, {"temperature": 0, "top_k": 20, "repetition_penalty": LG.LOOP_PENALTY,
                               "penalty_last_n": LG.PENALTY_MIN_N})
        self.assertEqual(n, 2 * 20 + LG.PENALTY_EXTRA)
        self.assertEqual(base, {"temperature": 0, "top_k": 20})    # (not changed in place)
        self.assertEqual(g.sampling(base, len(kept), 10)[1], 10)   # never past what is left
        self.assertIs(g.sampling(base, len(kept) + n, 500)[0], base)   # the window over: the request's own
        # a request's own stronger penalty or longer window stays
        smp, _ = g.sampling({"repetition_penalty": 1.3, "penalty_last_n": -1}, len(kept), 5000)
        self.assertEqual((smp["repetition_penalty"], smp["penalty_last_n"]), (1.3, -1))
        smp, _ = g.sampling({"penalty_last_n": 0}, len(kept), 5000)   # "no penalties": the guard's own window
        self.assertEqual(smp["penalty_last_n"], LG.PENALTY_MIN_N)
        self.assertEqual(LG.penalty_window(300), 600)


class Settings(unittest.TestCase):
    def test_on_by_default_and_the_ways_to_turn_it_off(self):
        self.assertTrue(loop_guard_setting({}, {}))
        self.assertFalse(loop_guard_setting({}, {"MAYA_LOOP_GUARD": "0"}))
        self.assertFalse(loop_guard_setting({}, {"STRATA_LOOP_GUARD": "off"}))
        self.assertFalse(loop_guard_setting({"loop_guard": False}, {}))
        self.assertFalse(loop_guard_setting({"loop_guard": "false"}, {}))
        self.assertTrue(loop_guard_setting({"loop_guard": True}, {}))
        self.assertTrue(loop_guard_setting({"loop_guard": False}, {"MAYA_LOOP_GUARD": "1"}))   # the environment wins

    def test_the_request_field(self):
        for value, want in ((True, True), (False, False), ("false", False), ("on", True), (0, False), (None, None)):
            req = {"loop_guard": value}
            check_request_sampling(req)
            self.assertEqual(req.get("loop_guard"), want)
        with self.assertRaises(ValueError):
            check_request_sampling({"loop_guard": "maybe"})


class Tok(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>"]


class LoopingEngine:
    """One answer, continued from wherever the prompt says it got to (the first prompt + what was written, as the
    server's restarts send it): `head`, then `block` again and again - unless a call samples with a repetition penalty
    and there is an `escape`: then that (and the end), and later calls go on with the rest of it; once the prompt holds
    the end of thinking, `answer` and the end.  Records every call: (prompt, max_new, sampling)."""

    def __init__(self, tok, head, block, escape=None, answer=None, think_end=None, script=None):
        self.tok, self.max_context = tok, 1 << 20
        self.end = tok.encode(IM_END, parse_special=True)
        enc = lambda s: tok.encode(s, parse_special=True) if s is not None else None   # noqa: E731
        self.head, self.block, self.escape, self.answer = enc(head), enc(block), enc(escape), enc(answer)
        self.script = enc(script)                   # (a plain answer, no loop: head/block unused)
        self.think_end, self.base, self.escaped_at = think_end, None, None
        self.calls: list[tuple[list[int], int, dict]] = []
        self.made = 0                               # tokens written in all (the streaming test reads it)

    def _at(self, k):
        return self.head[k] if k < len(self.head) else self.block[(k - len(self.head)) % len(self.block)]

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls.append((list(ids), max_new, dict(sampling or {})))
        if self.base is None:
            self.base = list(ids)
        written = ids[len(self.base):]
        rp = (sampling or {}).get("repetition_penalty") or 1.0
        if self.script is not None:
            out = self.script[len(written):] + self.end
        elif self.think_end is not None and self.think_end in written:
            out = self.answer + self.end
        elif self.escaped_at is not None and len(written) >= self.escaped_at:
            out = self.escape[len(written) - self.escaped_at:] + self.end
        elif self.escape is not None and rp > 1.0:
            self.escaped_at = len(written)
            out = self.escape + self.end
        else:
            out = (self._at(len(written) + i) for i in itertools.count())
        for i, t in enumerate(out):
            if i >= max_new or cancel.is_set():
                return
            self.made += 1
            yield t


HEAD = "Here are the numbers you asked for:\n"    # (its end is not the block's: the first copy stays as it is)
BLOCK = "1017, 245, 26, 100, 47, 187, "
ESCAPE = "and so on.\n\nIn short, the sequence repeats with period six."


def run(svc, max_new=3000, sampling=None, thinking=False, prompt="go"):
    """svc.run's stream: (reasoning text, content text, done, [(content chars so far, engine tokens then)])."""
    ids = svc.tok.encode(prompt)
    reasoning, content, seen, done = "", "", [], None
    for kind, val in svc.run(ids, thinking, None, max_new, dict(sampling or {}), threading.Event()):
        if kind == "event" and val.kind == "reasoning":
            reasoning += val.text
        elif kind == "event" and val.kind == "content":
            content += val.text
            seen.append((len(content), svc.engine.made))
        elif kind == "done":
            done = val
    return reasoning, content, done, seen


class Server(unittest.TestCase):
    def service(self, engine, tok=None):
        tok = tok or engine.tok
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        return svc

    def test_a_loop_is_dropped_and_penalised_then_escapes(self):
        eng = LoopingEngine(Tok(), HEAD, BLOCK, escape=ESCAPE)
        svc = self.service(eng)
        _, content, done, _ = run(svc, sampling={"temperature": 0})
        self.assertEqual(content, HEAD + BLOCK + ESCAPE)
        self.assertEqual(done["finish"], "stop")
        self.assertEqual(len(eng.calls), 2)
        (p1, n1, s1), (p2, n2, s2) = eng.calls
        self.assertEqual((n1, s1), (3000, {"temperature": 0}))     # the request's own until the loop
        period = len(BLOCK)
        self.assertEqual(p2, p1 + svc.tok.encode(HEAD + BLOCK))    # on from the first copy
        self.assertEqual(s2, {"temperature": 0, "repetition_penalty": LG.LOOP_PENALTY,
                              "penalty_last_n": max(LG.PENALTY_MIN_N, 2 * period)})
        self.assertEqual(n2, 2 * period + LG.PENALTY_EXTRA)
        m = svc.metrics()
        self.assertEqual({k: m["loop_guard"][k] for k in ("detected", "penalised", "stopped")},
                         {"detected": 1, "penalised": 1, "stopped": 0})
        self.assertTrue(m["loop_guard"]["on"])
        self.assertEqual(len(m["requests"][0]["loop_guard"]), 1)
        self.assertIn(f"a {period}-token block", m["requests"][0]["loop_guard"][0])
        self.assertIn("maya:loop_guard_detected_total 1", svc.prometheus())

    def test_the_request_sampling_comes_back_after_the_window(self):
        long_escape = " ".join(f"Step {i}: the term at {i} is {i * i % 97}." for i in range(30))
        eng = LoopingEngine(Tok(), HEAD, BLOCK, escape=long_escape)
        svc = self.service(eng)
        _, content, done, _ = run(svc, sampling={"temperature": 0.7, "seed": 5})
        self.assertEqual(content, HEAD + BLOCK + long_escape)
        self.assertEqual(done["finish"], "stop")
        self.assertEqual(len(eng.calls), 3)
        window = 2 * len(BLOCK) + LG.PENALTY_EXTRA
        self.assertEqual(eng.calls[1][1], window)
        self.assertEqual(eng.calls[2][2], {"temperature": 0.7, "seed": 5})   # the request's own again
        written = len(svc.tok.encode(HEAD + BLOCK)) + window
        self.assertEqual(eng.calls[2][1], 3000 - written)

    def test_a_loop_that_persists_in_the_answer_ends_it(self):
        eng = LoopingEngine(Tok(), HEAD, BLOCK)                     # loops whatever the penalty
        svc = self.service(eng)
        _, content, done, _ = run(svc)
        self.assertEqual(content, HEAD + BLOCK)
        self.assertEqual(done["finish"], "stop")
        self.assertLess(done["completion_tokens"], 200)
        self.assertEqual(len(eng.calls), 2)
        self.assertEqual(svc.loop_guard_counts, {"detected": 2, "penalised": 1, "thinking_closed": 0, "stopped": 1,
                                                 "tool_repeats": 0, "tool_blocked": 0, "collapsed": 0, "kept_going": 0})

    def test_a_second_loop_in_the_thinking_closes_it(self):
        tok = Tok()
        think_end = tok.encode("</think>", parse_special=True)[0]
        eng = LoopingEngine(tok, "I need to list them:\n", BLOCK, answer="\n\nThe numbers: 1017, 245, 26.",
                            think_end=think_end)
        svc = self.service(eng)
        self.assertEqual(svc.think_end_id, think_end)
        reasoning, content, done, _ = run(svc, thinking=True)
        self.assertEqual(reasoning, "I need to list them:\n" + BLOCK)
        self.assertEqual(content.strip(), "The numbers: 1017, 245, 26.")
        self.assertEqual(done["finish"], "stop")
        self.assertEqual(len(eng.calls), 3)
        self.assertEqual(eng.calls[2][0][-1], think_end)           # the thinking closed by its own token
        self.assertEqual(eng.calls[2][0].count(think_end), 1)
        self.assertEqual(svc.loop_guard_counts["thinking_closed"], 1)

    def test_streaming_holds_back_only_the_repeats(self):
        eng = LoopingEngine(Tok(), HEAD, BLOCK, escape=ESCAPE)
        svc = self.service(eng)
        _, content, _, seen = run(svc)
        self.assertEqual(content.count(BLOCK), 1)                  # a dropped copy never reached the client
        # what is not a repeat goes out as it comes: the head while the engine is still writing it
        n_head = len(svc.tok.encode(HEAD))
        early = [made for chars, made in seen if chars <= len(HEAD) - 4]
        self.assertTrue(early)
        self.assertLessEqual(max(made - chars for chars, made in seen if chars <= len(HEAD) - 4), 8)
        self.assertLessEqual(max(early), n_head + 4)

    def test_no_loop_same_calls_and_same_answer_on_and_off(self):
        # (shorter than LEGIT's: the tests' byte tokenizer decodes the whole answer again for every token)
        text = (numbers_table(12) + json.dumps(list(range(60))) + "\n" +
                "".join(f"result = process_item(items[{i}], options)\n" for i in range(12)) +
                "".join(f"    x[{i}] = a[{i}] * w[{i % 4}] + b[{i}];\n" for i in range(12)) +
                LEGIT["a block twice"] + LEGIT["a short line three times"])
        out = {}
        for mode in ("server on", "server off", "request off", "request on"):
            eng = LoopingEngine(Tok(), "", "", script=text)
            svc = self.service(eng)
            svc.loop_guard = mode != "server off"
            sampling = {"temperature": 0.6, "top_p": 0.95, "seed": 3}
            if mode.startswith("request"):
                svc.loop_guard = mode == "request off"              # (the request's field wins both ways)
                sampling["loop_guard"] = mode == "request on"
            r, c, done, _ = run(svc, max_new=20000, sampling=sampling)
            calls = [(p, n, {k: v for k, v in s.items() if k != "loop_guard"}) for p, n, s in eng.calls]
            out[mode] = (r, c, done["finish"], done["completion_tokens"], calls, svc.loop_guard_counts["detected"])
        self.assertEqual(out["server on"][1], text)
        for mode in ("server off", "request off", "request on"):
            self.assertEqual(out[mode], out["server on"], mode)
        self.assertEqual(len(out["server on"][4]), 1)              # one engine call, its sampling the request's

    def test_off_lets_the_loop_run(self):
        eng = LoopingEngine(Tok(), HEAD, BLOCK, escape=ESCAPE)
        svc = self.service(eng)
        svc.loop_guard = False
        _, content, done, _ = run(svc, max_new=600)
        self.assertEqual(done["finish"], "length")
        self.assertGreater(content.count(BLOCK), 15)
        self.assertEqual(len(eng.calls), 1)


class Http(unittest.TestCase):
    """The same through /v1/chat/completions: streamed deltas and the whole answer agree, one copy of the block."""

    @classmethod
    def setUpClass(cls):
        cls.eng = LoopingEngine(Tok(), HEAD, BLOCK, escape=ESCAPE)
        cls.svc = Service(cls.eng, cls.eng.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, body):
        self.eng.base, self.eng.escaped_at = None, None             # (a new conversation each time)
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read().decode()

    def test_whole_and_streamed(self):
        msg = {"messages": [{"role": "user", "content": "numbers"}], "max_tokens": 2000,
               "chat_template_kwargs": {"enable_thinking": False}}
        whole = json.loads(self.post(msg))
        self.assertEqual(whole["choices"][0]["message"]["content"], HEAD + BLOCK + ESCAPE)
        self.assertEqual(whole["choices"][0]["finish_reason"], "stop")
        text = ""
        for line in self.post(dict(msg, stream=True)).splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                for ch in json.loads(line[6:]).get("choices") or []:
                    text += (ch.get("delta") or {}).get("content") or ""
        self.assertEqual(text, HEAD + BLOCK + ESCAPE)
        off = json.loads(self.post(dict(msg, loop_guard=False, max_tokens=400)))
        self.assertEqual(off["choices"][0]["finish_reason"], "length")
        self.assertGreater(off["choices"][0]["message"]["content"].count(BLOCK), 8)


def agent(*rounds, user="find out why the module is offline"):
    """A Chat Completions request: the user's message, then (command, result) rounds of one bash call each."""
    msgs = [{"role": "user", "content": user}]
    for i, (cmd, out) in enumerate(rounds):
        msgs += [{"role": "assistant", "content": "", "tool_calls": [
                     {"id": f"c{i}", "type": "function",
                      "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]},
                 {"role": "tool", "tool_call_id": f"c{i}", "content": out}]
    return {"messages": msgs, "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}]}


NMCLI = ("nmcli -s connection show Smulnet | grep pmf", "802-11-wireless-security.pmf: 0 (default)")
GREP = ("grep -n UL_RETRY src/uplink.cpp", "13:#define UL_RETRY_MIN 2000")


class ToolRepeat(unittest.TestCase):
    """Across turns: the same tool call with the same result again and again (GLM-5.3 over opencode ran two
    commands in turn 25 times) gets a note after its result; the model reads it before it answers."""

    def messages(self, req):
        from serve.frontend import openai_to_messages
        return openai_to_messages(req)[0]

    def test_what_counts(self):
        rep = LG.tool_repeat(self.messages(agent(NMCLI, GREP, NMCLI, GREP, NMCLI)))
        self.assertEqual(rep, LG.ToolRepeat("bash", 3))
        self.assertIn("`bash` has now run 3 times", LG.tool_note(rep))
        self.assertIsNone(LG.tool_repeat(self.messages(agent(NMCLI, NMCLI, NMCLI)) +       # (the note itself)
                                         [{"role": "user", "content": LG.tool_note(rep)}]))
        self.assertIsNone(LG.tool_repeat(self.messages(agent(NMCLI, GREP, NMCLI, GREP))))       # twice: not yet
        # the same command after an edit: another result each time - not a loop
        edits = [r for n in (5, 4, 3, 2) for r in (("make test", f"{n} failed"), (f"sed -i {n}s/x/y/ a.c", ""))]
        self.assertIsNone(LG.tool_repeat(self.messages(agent(*edits[:-1]))))
        # in a row with nothing in between, a result that differs only in numbers or garbled bytes is the same one
        # (GLM-5.3's telnet test: "bytes received: 12 \ufffd\ufffd" then "15 \ufffd\ufffd\ufffd" ...) - not a wait
        probe = [("python3 probe.py", f"bytes received: {n} REPLY: " + "\ufffd" * n) for n in (12, 15, 12, 15)]
        self.assertEqual(LG.tool_repeat(self.messages(agent(*probe))), LG.ToolRepeat("bash", 4))
        self.assertIsNone(LG.tool_repeat(self.messages(agent(*[("sleep 30; tail -1 train.log", f"step {n}00/900")
                                                               for n in range(1, 6)]))))
        # the user's "continue" says nothing new: the count goes on
        req = agent(NMCLI, NMCLI)
        req["messages"] += [{"role": "user", "content": "Continue."}] + agent(NMCLI)["messages"][1:]
        self.assertEqual(LG.tool_repeat(self.messages(req)), LG.ToolRepeat("bash", 3))
        # the user spoke since: only the calls after their message count
        req = agent(NMCLI, NMCLI, NMCLI)
        req["messages"] += [{"role": "user", "content": "stop repeating and try a different approach"}]
        req["messages"] += agent(NMCLI)["messages"][1:]
        self.assertIsNone(LG.tool_repeat(self.messages(req)))
        # the request does not end with a tool result: the model is answering the user
        self.assertIsNone(LG.tool_repeat(self.messages(agent(NMCLI, NMCLI, NMCLI, user="x")) +
                                         [{"role": "user", "content": "thanks"}]))

    def test_the_prompt_gets_the_note_and_the_counts(self):
        svc = Server().service(LoopingEngine(Tok(), HEAD, BLOCK))
        seen = []
        render = svc.render_ids
        svc.render_ids = lambda messages, *a, **k: seen.append(messages) or render(messages, *a, **k)
        req = agent(NMCLI, GREP, NMCLI, GREP, NMCLI)
        msgs, tools, kw = __import__("serve.frontend", fromlist=["x"]).openai_to_messages(req)
        svc.prepare(msgs, tools, dict(kw), 100)
        self.assertEqual(seen[-1][-1], {"role": "user", "content": LG.tool_note(LG.ToolRepeat("bash", 3))})
        self.assertEqual(seen[-1][:-1], LG.collapse(msgs).messages)     # (the repeated rounds: left out, below)
        self.assertEqual(msgs[-1]["content"], NMCLI[1])                 # the request's own messages stay as sent
        self.assertEqual(svc.metrics()["loop_guard"]["tool_repeats"], 1)
        self.assertIn("maya:loop_guard_tool_repeats_total 1", svc.prometheus())
        svc.prepare(msgs, tools, dict(kw), 100, guard=False)          # the request's "loop_guard": false
        self.assertEqual(seen[-1], msgs)
        svc.prepare(self.messages(agent(NMCLI, GREP)), tools, dict(kw), 100)
        self.assertEqual(seen[-1][-1], {"role": "tool", "content": GREP[1]})
        self.assertEqual(svc.metrics()["loop_guard"]["tool_repeats"], 1)


def call_text(cmd):
    return f"<tool_call>\n<function=bash>\n<parameter=command>\n{cmd}\n</parameter>\n</function>\n</tool_call>"


class ScriptEngine:
    """Answer k of a request (its k-th engine call) is an empty thinking and scripts[k] (the last one again after
    that), then the end.  Records every call's prompt."""

    def __init__(self, tok, *scripts):
        self.tok, self.max_context, self.made = tok, 1 << 20, 0
        self.end = tok.encode(IM_END, parse_special=True)
        self.scripts = [tok.encode("</think>\n\n" + s, parse_special=True) for s in scripts]   # (thinking: on)
        self.prompts: list[list[int]] = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        for t in (self.scripts[min(len(self.prompts), len(self.scripts)) - 1] + self.end)[:max_new]:
            self.made += 1
            yield t


class ToolBlock(unittest.TestCase):
    """After TOOL_NUDGES notes the model wrote the same call again (GLM-5.3: "the loop guard is right, act
    differently" - and the same grep, ten more times): that call is blocked - it never reaches the client, the server
    answers it in the prompt and the model goes on in the same answer."""

    def answer(self, *scripts, rounds=(NMCLI,) * 5, engine=None):
        from serve.frontend import openai_to_messages
        eng = engine or ScriptEngine(Tok(), *scripts)
        svc = Server().service(eng)
        msgs, tools, kw = openai_to_messages(agent(*rounds))
        ids, thinking, max_new = svc.prepare(msgs, tools, kw, 2000)
        events, done = [], None
        for kind, x in svc.run(ids, thinking, tools, max_new, {"_blocked": kw.get("_blocked")}, threading.Event()):
            if kind == "event":
                events.append(x)
            elif kind == "done":
                done = x
        calls = [e.call.arguments["command"] for e in events if e.kind == "tool_call"]
        text = "".join(e.text for e in events if e.kind == "content")
        return svc, eng, kw, calls, text, done

    def test_notes_differ_and_when_it_blocks(self):
        notes = [LG.tool_note(LG.ToolRepeat("bash", t)) for t in range(LG.TOOL_REPS, LG.TOOL_BLOCK + 1)]
        self.assertEqual(len(notes), LG.TOOL_NUDGES)
        self.assertEqual(len(set(notes)), LG.TOOL_NUDGES)               # worded differently each time
        self.assertIn("blocked", notes[-1])                             # the last one says what comes next
        msgs = ToolRepeat().messages
        self.assertEqual(LG.tool_blocked(msgs(agent(NMCLI, GREP, NMCLI, GREP, NMCLI, GREP, NMCLI))), {})   # 4: notes
        got = LG.tool_blocked(msgs(agent(NMCLI, GREP, NMCLI, GREP, NMCLI, GREP, NMCLI, GREP, NMCLI)))
        self.assertEqual(list(got.values()), [LG.ToolRepeat("bash", 5)])   # NMCLI 5 times; GREP 4: not yet
        self.assertEqual(list(got), [("bash", LG.args_key({"command": NMCLI[0]}))])
        req = agent(*(NMCLI,) * 5)
        req["messages"].append({"role": "user", "content": "try the band"})
        self.assertEqual(LG.tool_blocked(msgs(req)), {})                 # the user spoke: free again

    def test_blocked_then_moves_on(self):
        other = "nmcli -f 802-11-wireless-security.key-mgmt connection show Smulnet"
        svc, eng, kw, calls, text, done = self.answer(call_text(NMCLI[0]), "Checking the key management.\n" +
                                                      call_text(other))
        self.assertEqual(calls, [other])                                # the blocked one never went out
        self.assertEqual(len(eng.prompts), 2)
        first, second = eng.prompts
        said = svc.tok.decode(second[len(first):])
        self.assertTrue(said.startswith("</think>\n\n" + call_text(NMCLI[0])))   # what the model wrote, kept ...
        self.assertIn(LG.block_result(LG.ToolRepeat("bash", 5)), said)  # ... the server's answer to it ...
        self.assertIn(LG.block_note(LG.ToolRepeat("bash", 5)), said)    # ... and the note
        self.assertTrue(second[:len(first)] == first)                   # (the conversation's cache goes on)
        self.assertEqual((done["finish"], svc.metrics()["loop_guard"]["tool_blocked"]), ("stop", 1))
        self.assertIn("maya:loop_guard_tool_blocked_total 1", svc.prometheus())

    def test_again_and_again_it_is_steered_never_stopped(self):
        other = "cat host/bellhost/bridge.py"
        eng = ScriptEngine(Tok(), *[call_text(NMCLI[0])] * 3, call_text(other))
        eng.temps = []
        gen = eng.generate
        eng.generate = lambda ids, n, smp, *a, **k: eng.temps.append((smp or {}).get("temperature")) or gen(ids, n, smp,
                                                                                                         *a, **k)
        svc, eng2, kw, calls, text, done = self.answer(engine=eng)
        self.assertEqual(calls, [other])                                # it went on until it did something else
        self.assertNotIn("stopped", text)
        self.assertEqual(len(eng.prompts), 4)
        rep, args = LG.ToolRepeat("bash", 5), LG.args_key({"command": NMCLI[0]})
        for k in (1, 2, 3):                                             # each time its thinking starts elsewhere
            said = svc.tok.decode(eng.prompts[k][len(eng.prompts[k - 1]):])
            self.assertTrue(said.endswith(LG.block_steer(rep, args, k)), said[-200:])
        self.assertEqual(len({LG.block_steer(rep, args, k) for k in (1, 2, 3)}), 3)
        self.assertEqual(eng.temps[:2], [None, None])                  # the request's own, then hotter from the 2nd
        self.assertGreater(eng.temps[3], eng.temps[2])
        self.assertLessEqual(max(t for t in eng.temps if t), LG.BLOCK_HEAT_MAX)
        self.assertEqual(svc.metrics()["loop_guard"]["tool_blocked"], 3)

    def test_a_continue_hears_what_is_blocked(self):
        from serve.frontend import openai_to_messages
        svc = Server().service(ScriptEngine(Tok(), "x"))
        seen, render = [], svc.render_ids
        svc.render_ids = lambda messages, *a, **k: seen.append(messages) or render(messages, *a, **k)
        req = agent(*(NMCLI,) * 5)
        req["messages"].append({"role": "user", "content": "continue"})
        msgs, tools, kw = openai_to_messages(req)
        svc.prepare(msgs, tools, kw, 100)
        self.assertTrue(kw["_blocked"])                                 # still blocked after "continue" ...
        self.assertEqual(seen[-1][-2:], [msgs[-1], {"role": "user", "content": LG.blocked_note(
            LG.tool_blocked(msgs))}])                                   # ... and the model is told so first
        self.assertIn(NMCLI[0], seen[-1][-1]["content"])

    def test_it_does_not_end_its_turn_with_nothing(self):
        # GLM-5.3 ended its turn with "OK - the loop guard is right. ... Break out: use what I have or answer the
        # user." and the agent stopped: in a conversation that repeats itself, that answer goes on
        other = "grep -n on_bytes host/bellhost/bridge.py"
        talk = "OK \u2014 the loop guard is right. Stop repeating identical greps. Break out: use what I have."
        svc, eng, kw, calls, text, done = self.answer(talk, call_text(other), rounds=(NMCLI,) * 4)
        self.assertEqual(calls, [other])
        said = svc.tok.decode(eng.prompts[1][len(eng.prompts[0]):])
        self.assertIn(LG.KEEP_NOTE, said)
        self.assertTrue(said.endswith(LG.keep_steer(1)))
        self.assertEqual(svc.metrics()["loop_guard"]["kept_going"], 1)
        # a report of what it found may end the turn
        report = "What I found:\n\n" + "\n".join(f"- Finding {i}: the operator claim at step {i} is read-only, so "
                                                  "the bytes are dropped in on_bytes." for i in range(5))
        svc, eng, kw, calls, text, done = self.answer(report, call_text(other), rounds=(NMCLI,) * 4)
        self.assertEqual((calls, len(eng.prompts)), ([], 1))
        # a conversation that does not repeat itself: a short answer ends the turn as always
        svc, eng, kw, calls, text, done = self.answer("Done.", call_text(other), rounds=(NMCLI, GREP))
        self.assertEqual((calls, len(eng.prompts), text), ([], 1, "Done."))

    def test_what_is_not_blocked(self):
        # four runs: notes only, the call goes out
        self.assertEqual(self.answer(call_text(NMCLI[0]), rounds=(NMCLI,) * 4)[3], [NMCLI[0]])
        # other arguments, and a blocked call next to another one in the same answer
        self.assertEqual(self.answer(call_text("nmcli dev wifi"))[3], ["nmcli dev wifi"])
        self.assertEqual(self.answer(call_text("ls") + "\n" + call_text(NMCLI[0]))[3], ["ls", NMCLI[0]])
        # the request's "loop_guard": false: nothing blocked
        from serve.frontend import openai_to_messages
        svc = Server().service(ScriptEngine(Tok(), "x"))
        msgs, tools, kw = openai_to_messages(agent(*(NMCLI,) * 5))
        svc.prepare(msgs, tools, kw, 100, guard=False)
        self.assertIsNone(kw["_blocked"])


class Collapse(unittest.TestCase):
    """A conversation that repeats itself: the prompt leaves out what the model already wrote (GLM-5.3 had "Hmm -
    hold on: I realize something important." 543 times in an 88K prompt and went on writing it)."""

    STUCK = ("Hmm \u2014 hold on: I realize something important. Hmm \u2014 hmm.\n\n"
             "OK let me stop theorizing and go empirical: check the slot log.")

    def turns(self, n, new=lambda i: ""):
        """The user's question, then n rounds of the same thinking and the same grep (round i's own thinking new(i))."""
        msgs = [{"role": "user", "content": "why do nc bytes never reach the device?"}]
        for i in range(n):
            msgs += [{"role": "assistant", "content": "", "reasoning_content": self.STUCK + new(i),
                      "tool_calls": [{"id": f"c{i}", "type": "function", "function": {
                          "name": "bash", "arguments": {"command": NMCLI[0]}}}]},
                     {"role": "tool", "tool_call_id": f"c{i}", "content": NMCLI[1]}]
        return msgs

    def test_what_is_left_out(self):
        self.assertIsNone(LG.collapse(self.turns(2)))                  # twice: no loop yet, the prompt as it is
        msgs = self.turns(4, new=lambda i: f" Found: claim {i} is read-only." if i == 2 else "")
        c = LG.collapse(msgs)
        self.assertEqual(len(c.messages), 1 + 2)                       # the 1st round, whole; the rest repeats it -
        self.assertEqual(c.messages[1]["reasoning_content"],          # its thinking with it, less the talk about
                         "Hmm \u2014 hold on: I realize something important. Hmm \u2014 hmm.")   # the loop
        self.assertEqual(c.rounds, 3)
        msgs[5]["content"] = "The claim is read-only, so nc's bytes are dropped."   # text for the user stays
        self.assertEqual(LG.collapse(msgs).messages[3], {"role": "assistant", "content": msgs[5]["content"]})
        self.assertEqual(msgs[3]["reasoning_content"], self.STUCK)      # the request's own messages stay as sent

    def test_a_check_run_again_in_a_row(self):
        msgs = [{"role": "user", "content": "x"}]
        for i, n in enumerate((12, 15, 12, 15)):
            msgs += [{"role": "assistant", "content": "", "reasoning_content": f"Try {i}.", "tool_calls": [
                         {"id": f"c{i}", "type": "function", "function": {"name": "bash",
                                                                          "arguments": {"command": "probe"}}}]},
                     {"role": "tool", "tool_call_id": f"c{i}", "content": f"bytes received: {n}"}]
        c = LG.collapse(msgs)
        self.assertEqual(len(c.messages), 3)                           # the first; the rest differ only in numbers

    def test_the_next_request_starts_the_same(self):
        msgs = self.turns(8, new=lambda i: f" Step {i}." if i % 3 == 0 else "")
        whole = LG.collapse(msgs).messages
        for cut in range(3, len(msgs) + 1, 2):                         # (every request ends with a tool result)
            part = LG.collapse(msgs[:cut])
            if part is not None:
                self.assertEqual(part.messages, whole[:len(part.messages)])

    def test_the_answer_text_by_paragraph_and_short_ones_stay(self):
        code = "```\n}\n```"
        msgs = [{"role": "user", "content": "x"}] + [{"role": "assistant", "content": f"Done.\n\n{code}\n\nThe fix "
                                                      "is in uplink.cpp, line 13."} for _ in range(3)]
        c = LG.collapse(msgs)
        self.assertEqual([m["content"] for m in c.messages[2:]], [f"Done.\n\n{code}"] * 2)

    def test_a_conversation_without_a_loop_is_untouched(self):
        msgs = [{"role": "user", "content": "x"}]
        for i in range(6):
            msgs += [{"role": "assistant", "content": f"Step {i}: reading part {i}.", "reasoning_content": f"Part {i}."},
                     {"role": "user", "content": f"ok {i}"}]
        self.assertIsNone(LG.collapse(msgs))


class GuardedUsage(unittest.TestCase):
    """After the guard, the engine goes on from prompt + the kept tokens: the request reports its own prompt's
    figures, not that call's (a guarded MiniMax answer said 5,919 cached tokens of a 246-token prompt)."""

    def test_the_request_prompt_not_the_resumed_call(self):
        from serve.server import own_prompt_figures
        got = own_prompt_figures({"prompt_tokens": 246 + 5546, "reused": 5546 + 127, "prompt_ms": 9.5}, 246)
        self.assertEqual((got["prompt_tokens"], got["reused"], got["prompt_ms"]), (246, 127, 9.5))
        self.assertEqual(own_prompt_figures({"prompt_tokens": 1300, "reused": 300}, 1000)["reused"], 0)
        self.assertEqual(own_prompt_figures({"prompt_tokens": 1300, "reused": 5000}, 1000)["reused"], 1000)


if __name__ == "__main__":
    unittest.main()
