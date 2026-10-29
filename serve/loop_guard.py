"""serve/loop_guard.py - the loop guard: an answer that has fallen into a loop - the same block of tokens again and
again until the token cap (a greedy MiniMax-M3.1 Maya-XS answer wrote one number sequence for 12K tokens) - is seen
as it starts and broken.  It works on the token ids the server reads from the engine, so every engine (GLM,
DeepSeek, MiniMax) gets it alike, in the thinking and in the answer.

    LoopDetector   the rolling check: is the tail of what was written exactly periodic - a block of P_MIN..P_MAX
                   tokens repeated back to back REPS_LONG times (REPS_SHORT for a block under SHORT_BELOW tokens),
                   MIN_SPAN tokens at least?  Each token is compared only with the earlier places of the same id
                   (within P_MAX), so a token costs microseconds.
    LoopGuard      one request's guard: the detector, the tail the stream holds back (the copies a loop's repeats
                   would be are never sent before it is clear they are not one), and what each loop does:
                     1st              the repeats are dropped (the first copy stays) and the engine goes on from
                                      there with a short repetition penalty (LOOP_PENALTY over the last 2 x period
                                      tokens, at least PENALTY_MIN_N, for the next 2 x period + PENALTY_EXTRA)
                     2nd, thinking    the thinking is closed with the model's own end-of-thinking token: it answers
                     2nd in the answer, or a 3rd    the answer ends there (finish "stop")

Across turns, an agent can loop with every answer short and fine: it runs the same tool call, gets the same result
and runs it again (GLM-5.3 ran two commands in turn 25 times over opencode, until the user wrote "stop repeating").

    tool_repeat      the request's own conversation: has the last tool call - its name, its arguments and its result
                     all the same - come TOOL_REPS times among the last TOOL_WINDOW calls since the user's last
                     message?  Then tool_note() follows that result as a message of its own, for the model to read before it
                     answers (inside the tool's result GLM-5.3 ran the call a 4th time; as a message it moved on).

Until a loop is seen nothing changes: the engine gets the same calls with the same sampling and writes the same
tokens (the stream only holds back the tail that repeats while it does).  Legit repetition - a table's rows, an array
literal, a function name over and over, code lines alike - is not an exact block repeated back to back three times
or more, which is what a loop is.

Off: MAYA_LOOP_GUARD=0, "loop_guard": false in the run config, or "loop_guard": false in a request (true there turns
it on for that request when the server has it off).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import NamedTuple

P_MIN = 8               # the shortest block: shorter loops (one token, "0, 0, 0") repeat a multiple of it in 8..15
P_MAX = 2048            # the longest: a loop of whole paragraphs or a plan written again (Maya-S thought a ~1,170-
                        # token plan four times); answers without a loop hold back and cost the same as with 400
SHORT_BELOW = 32        # blocks under this many tokens must repeat REPS_SHORT times, longer ones REPS_LONG times
REPS_SHORT = 4
REPS_LONG = 3
MIN_SPAN = 96           # ... and the copies together cover this many tokens at least
LOOP_PENALTY = 1.15     # the temporary repetition penalty after the first loop
PENALTY_MIN_N = 128     # its window (penalty_last_n): 2 x the period, at least this
PENALTY_EXTRA = 128     # it applies to the next 2 x period + this many tokens, then the request's own sampling again
PENALTY_N_MAX = 4096    # (the engines count at most this many: server.PENALTY_WINDOW_MAX)

PENALTY, CLOSE_THINKING, STOP = "penalty", "close_thinking", "stop"


class Loop(NamedTuple):
    """A loop at the tail: `period` tokens a copy; the last `repeats` tokens repeat the `period` before them."""
    period: int
    repeats: int

    @property
    def copies(self) -> float:
        return (self.repeats + self.period) / self.period


class Action(NamedTuple):
    kind: str           # PENALTY, CLOSE_THINKING or STOP
    period: int
    copies: float
    drop: int           # the repeats to drop from the end (the first copy stays)
    where: str          # "thinking" or "answer"
    at: int             # the tokens written when it was seen (before the drop)

    def describe(self) -> str:
        what = {PENALTY: "repeats dropped, a short repetition penalty", CLOSE_THINKING: "thinking closed",
                STOP: "answer ended"}[self.kind]
        return (f"a {self.period}-token block {self.copies:.0f} times in the {self.where} at token {self.at:,}: "
                f"{what}")


class LoopDetector:
    """The tail's exact periodicity, updated a token at a time.  `runs[p]` is how many tokens in a row (to the last
    one) equal the token p before them; p is a loop's period once its run covers the copies it needs (`need[p]`).
    Only the periods where the new token equals the one p back go on (the earlier places of that id within P_MAX),
    every other run ends - so the work a token is the number of times its id occurred in the last P_MAX tokens.
    `hold` is the longest run now: the tokens that could still turn out to be a loop's repeats."""

    def __init__(self, p_min: int = P_MIN, p_max: int = P_MAX, min_span: int = MIN_SPAN,
                 reps_short: int = REPS_SHORT, reps_long: int = REPS_LONG, short_below: int = SHORT_BELOW):
        self.p_min, self.p_max = p_min, p_max
        self.need = [0] * (p_max + 1)
        for p in range(p_min, p_max + 1):
            reps = reps_short if p < short_below else reps_long
            self.need[p] = max((reps - 1) * p, min_span - p)
        self.reset()

    def reset(self, tokens=()) -> None:
        """Start again over `tokens` (what is kept after a loop's repeats were dropped): only their last 3 x P_MAX
        count - a run reaches at most 2 x P_MAX before it is a loop."""
        self.n, self.seen, self.runs, self.hold = 0, {}, {}, 0
        tokens = list(tokens)[-3 * self.p_max:]
        for t in tokens:
            self.push(t)                        # (a prefix of what was checked token by token: no loop in it)

    def push(self, t: int) -> Loop | None:
        i = self.n
        self.n = i + 1
        q = self.seen.get(t)
        runs, hold, best = {}, 0, 0
        if q is None:
            self.seen[t] = [i]
        else:
            lo, k = i - self.p_max, 0
            while k < len(q) and q[k] < lo:
                k += 1
            if k:
                del q[:k]
            old, need, p_min = self.runs, self.need, self.p_min
            for j in q:                         # (oldest first: the periods from the longest down)
                p = i - j
                if p < p_min:
                    break
                r = old.get(p, 0) + 1
                runs[p] = r
                if r > hold:
                    hold = r
                if r >= need[p]:
                    best = p                    # the shortest that is a loop: its fundamental period
            q.append(i)
        self.runs, self.hold = runs, hold
        return Loop(best, runs[best]) if best else None


def penalty_window(period: int) -> int:
    return min(PENALTY_N_MAX, max(PENALTY_MIN_N, 2 * period))


class LoopGuard:
    """One request's loop guard (server.Service._sliced runs it): push() every id the engine writes, send only what
    lies before the last `hold` of them, and do what an Action says: drop its repeats, close the thinking (append
    `think_end`), or end; then resume() with what is kept and sampling() for each engine call after it."""

    def __init__(self, think_end: int | None = None, in_think: bool = False, detector: LoopDetector | None = None):
        self.det = detector or LoopDetector()
        self.think_end = think_end
        self.in_think = bool(in_think) and think_end is not None
        self.hits = {"thinking": 0, "answer": 0}
        self.events: list[Action] = []
        self.until = 0                          # the tokens written up to which the temporary penalty applies
        self.window = 0                         # ... and its penalty_last_n

    @property
    def hold(self) -> int:
        return self.det.hold

    def push(self, t: int, written: int | None = None) -> Action | None:
        if self.in_think and t == self.think_end:
            self.in_think = False
        loop = self.det.push(t)
        if loop is None:
            return None
        where = "thinking" if self.in_think else "answer"
        self.hits[where] += 1
        if sum(self.hits.values()) >= 3:
            kind = STOP
        elif where == "thinking":
            kind = PENALTY if self.hits[where] == 1 else CLOSE_THINKING
        else:
            kind = PENALTY if self.hits[where] == 1 else STOP
        act = Action(kind, loop.period, loop.copies, loop.repeats, where, written if written is not None else 0)
        self.events.append(act)
        return act

    def resume(self, kept: list[int], act: Action) -> None:
        """After the repeats were dropped (and the thinking closed): check on from what is kept, with the temporary
        penalty for the next 2 x period + PENALTY_EXTRA tokens."""
        if act.kind == CLOSE_THINKING:
            self.in_think = False
        self.det.reset(kept)
        if act.kind != STOP:
            self.window = penalty_window(act.period)
            self.until = len(kept) + 2 * act.period + PENALTY_EXTRA

    def sampling(self, base: dict | None, written: int, left: int) -> tuple[dict | None, int]:
        """The sampling for an engine call that starts after `written` tokens, and how many tokens it may write:
        the request's own (and `left`) - unless the temporary penalty runs, which then ends that call where it ends.
        A request's own stronger penalty or longer window stays."""
        if written >= self.until:
            return base, left
        smp = dict(base or {})
        rp = smp.get("repetition_penalty")
        rp = float(rp) if isinstance(rp, (int, float)) and not isinstance(rp, bool) else 1.0
        smp["repetition_penalty"] = max(rp, LOOP_PENALTY)
        pln = smp.get("penalty_last_n")
        if not (isinstance(pln, int) and not isinstance(pln, bool) and (pln == -1 or pln >= self.window)):
            smp["penalty_last_n"] = self.window
        return smp, max(1, min(left, self.until - written))

    def summary(self) -> list[str]:
        return [a.describe() for a in self.events]


# ------------------------------------------------------------------------------------------------ across turns
TOOL_WINDOW = 12        # the last this many tool calls since the user's last message ...
TOOL_REPS = 3           # ... hold the last one (name, arguments and result the same) this many times: a loop
TOOL_NUDGES = 3         # notes (tool_note) before the server stops running it: then the call itself is blocked
TOOL_BLOCK = TOOL_REPS + TOOL_NUDGES - 1   # (made this many times: TOOL_NUDGES notes went by, unheeded)
BLOCK_TRIES = 2         # blocked attempts in one answer the model gets a block note for; the one after ends it


class ToolRepeat(NamedTuple):
    """The last tool call of a conversation that keeps coming back: `name`, `times` among the last TOOL_WINDOW."""
    name: str
    times: int


def _same(v) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)


def args_key(v) -> str:
    """A call's arguments in one form: a client's JSON string and the parser's dict compare equal."""
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            pass
    return _same(v)


def _tool_calls(messages: list[dict]) -> list[tuple]:
    """(name, arguments, result) of each tool call since the user's last message, in order."""
    calls, pending = [], []
    for m in messages:
        role = m.get("role")
        if role == "user":                      # the user spoke: what came before is not this loop
            calls, pending = [], []
        elif role == "assistant":
            pending = [((c.get("function") or c).get("name"), args_key((c.get("function") or c).get("arguments")))
                       for c in m.get("tool_calls") or []]
        elif role == "tool" and pending:        # results answer the calls in their order
            name, args = pending.pop(0)
            calls.append((name, args, _same(m.get("content"))))
    return calls


def tool_repeat(messages: list[dict]) -> ToolRepeat | None:
    """The template messages (frontend.openai_to_messages / anthropic_to_messages) of a request that ends with tool
    results: is the last call one the model made TOOL_REPS times among its last TOOL_WINDOW since the user's last
    message, with the same arguments, and got the same result each time?  A command run again after an edit gives
    another result, a poll that finally changes another one too: neither counts."""
    if not messages or messages[-1].get("role") != "tool":
        return None
    calls = _tool_calls(messages)
    if not calls:
        return None
    times = calls[-TOOL_WINDOW:].count(calls[-1])
    return ToolRepeat(str(calls[-1][0]), times) if times >= TOOL_REPS else None


def tool_blocked(messages: list[dict]) -> dict[tuple[str, str], ToolRepeat]:
    """The calls this request's answer may not make again, by (name, args_key(arguments)): each made TOOL_BLOCK times
    among the last TOOL_WINDOW since the user's last message with the same arguments and result - the notes after
    TOOL_REPS of them did not stop it.  The model may still run anything else, these too with other arguments."""
    window = _tool_calls(messages)[-TOOL_WINDOW:]
    return {(str(c[0]), c[1]): ToolRepeat(str(c[0]), window.count(c)) for c in set(window)
            if window.count(c) >= TOOL_BLOCK}


def block_result(rep: ToolRepeat) -> str:
    """The tool result the server writes for a blocked call (the client never sees the call)."""
    return (f"[Not run. The server's loop guard blocked this call: `{rep.name}` already ran {rep.times} times since "
            "the user's last message with these same arguments, and gave the same result every time.]")


def block_note(rep: ToolRepeat) -> str:
    """The message after a blocked call's result: firmer than tool_note, since that one went unheeded."""
    return (f"[The server's loop guard, not the user: you have been told {TOOL_NUDGES} times that `{rep.name}` with "
            "these arguments gives nothing new, and it is now blocked: it will not run again in this answer. Do not "
            "write it again. Change something real - other arguments, another command or tool, another part of the "
            "problem - or stop and tell the user what you found and what is blocking you.]")


def block_stop(rep: ToolRepeat) -> str:
    """The answer's last words when the model wrote a blocked call again and again in one answer: back to the user."""
    return (f"\n\n[The server's loop guard stopped here: the model kept running `{rep.name}` with the same arguments "
            f"after it was blocked ({rep.times} runs, all with the same result). Tell it what to do next.]")


NOTES = (    # tool_note's wording, one after another: the same words each time are easy to read past
    "[The server's loop guard, not the user: `{name}` has now run {times} times since the user's last message with "
    "these same arguments and given this same result. Running it again will not show anything new. Do not repeat it: "
    "use what you already have, try a different approach, or answer the user.]",
    "[Loop guard (the server, not the user): that was `{name}` again - {times} runs, identical arguments, identical "
    "output. The output is not going to change. Pick something else: a different command, a different file, a test "
    "that can actually fail, or write up what you know for the user.]",
    "[Server loop guard, last warning: `{name}` with these arguments has given the same result {times} times. One more "
    "identical call will be blocked and not run. Stop checking this; act on what it already told you, or explain to "
    "the user where you are stuck.]",
)


def tool_note(rep: ToolRepeat) -> str:
    """The message after the repeated call's result: plain words, as a user would write them, said to be the
    server's - worded differently each time (NOTES, by how many runs; the last says a block comes next)."""
    k = min(max(rep.times - TOOL_REPS, 0), len(NOTES) - 1)
    return NOTES[k].format(name=rep.name, times=rep.times)


# ------------------------------------------------------------------------------------------------ what it already wrote
ECHO_TURNS = 3          # a paragraph in this many of the model's messages (or a call + result made this many times) ...
ECHO_MIN = {"reasoning_content": 6, "content": 12}   # ... (pieces this long or longer: in the answer's text
#                         shorter ones - "}", "Done." - are left alone; in the reasoning only "OK." and the like) ...
#                         the conversation carries a loop: the prompt leaves out what the model already wrote (collapse)


class Collapse(NamedTuple):
    """collapse()'s prompt and what it left out of it."""
    messages: list[dict]
    paragraphs: int     # sentences of its reasoning and paragraphs of its text the model had already written
    rounds: int         # tool rounds whose calls and results all repeat earlier ones


def _units(text, field: str) -> list[tuple[str, str]]:
    """(piece, the separator after it): the reasoning by sentence and line (prose; GLM-5.3 also runs its sentences
    together, "Hmm - hmm.Now let me ..."), the answer's text by paragraph (it may hold code)."""
    if not isinstance(text, str):
        return []
    sep = r"(\n+|(?<=[.!?])\s*(?=[A-Z*`\[(]))" if field == "reasoning_content" else r"(\n[ \t]*\n)"
    parts = re.split(sep, text)
    return [(parts[k], parts[k + 1] if k + 1 < len(parts) else "") for k in range(0, len(parts), 2)]


def _key(paragraph: str) -> str:
    return " ".join(paragraph.split())


def _round_keys(messages: list[dict], i: int) -> tuple[list[tuple], int]:
    """The (name, arguments, result) of messages[i]'s calls and the index after its results."""
    calls = [((c.get("function") or c).get("name"), args_key((c.get("function") or c).get("arguments")))
             for c in messages[i].get("tool_calls") or []]
    j, keys = i + 1, []
    while j < len(messages) and messages[j].get("role") == "tool" and len(keys) < len(calls):
        keys.append(calls[len(keys)] + (_same(messages[j].get("content")),))
        j += 1
    return keys, j


def echoes(messages: list[dict]) -> bool:
    """Does the conversation carry a loop across turns: one paragraph in ECHO_TURNS of the model's messages, or one
    call made ECHO_TURNS times with the same arguments and result?  (It only grows as the conversation does: once
    true it stays true, so the prompt collapse() makes stays the same prefix from one request to the next.)"""
    para, calls = Counter(), Counter()
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        para.update({_key(p) for f in ("reasoning_content", "content") for p, _ in _units(m.get(f), f)
                     if len(_key(p)) >= ECHO_MIN[f]})
        calls.update(_round_keys(messages, i)[0])
    return any(n >= ECHO_TURNS for n in para.values()) or any(n >= ECHO_TURNS for n in calls.values())


def collapse(messages: list[dict]) -> Collapse | None:
    """The prompt of a conversation that carries a loop (echoes()) without what the model already wrote: a paragraph
    of its reasoning or text that it wrote before (an earlier message, or earlier in the same one) is left out, and
    a round whose calls and results all repeat earlier ones, with nothing new said, goes with its results.  What the
    model sees is then what it found, once - not dozens of copies of the same thoughts to go on from (GLM-5.3 had
    "Hmm - hold on: I realize something important." 543 times in an 88K prompt, 89% of its own text repeats).  Each
    message depends only on the ones before it, so the next request's prompt starts the same.  None: no loop, or
    nothing to leave out."""
    if not echoes(messages):
        return None
    out, seen, made, paras, rounds, i = [], set(), set(), 0, 0, 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") != "assistant":
            out.append(m)
            i += 1
            continue
        m = dict(m)
        for f in ("reasoning_content", "content"):
            if not isinstance(m.get(f), str):
                continue
            kept = []
            for p, sep in _units(m[f], f):
                k = _key(p)
                if len(k) >= ECHO_MIN[f] and k in seen:
                    paras += 1
                    continue
                seen.add(k)
                kept.append(p + sep)
            m[f] = "".join(kept).strip()
        keys, j = _round_keys(messages, i)
        said = (m.get("content") or "").strip() or (m.get("reasoning_content") or "").strip()
        if keys and len(keys) == len(m.get("tool_calls") or []) and all(k in made for k in keys) and not said:
            rounds += 1
            i = j                                   # the round and its results
            continue
        made.update(keys)
        if said or m.get("tool_calls"):
            out.append(m)
        i += 1
    return Collapse(out, paras, rounds) if paras or rounds else None
