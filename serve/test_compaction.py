"""serve/test_compaction.py - a request to compact a conversation that is too long for the context
(serve/compaction.py): the oldest part of the conversation it quotes is left out until it fits; nothing else is.

    python -m unittest serve.test_compaction -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import compaction as C  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, ContextOverflow, Service  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASK = "Create a new anchored summary from the conversation history in the <conversation> tags above."


class Engine:
    def __init__(self, tok, max_context):
        self.tok, self.max_context = tok, max_context


def history(turns):
    return "\n".join(f"[User]: step {k}\n[Assistant tool call]: bash({{\"command\": \"make {k}\"}})\n"
                     f"[Tool result]: {'built ' * 20}{k}" for k in range(turns))


def compact(turns, parts=False):
    text = f"Here is the conversation so far:\n{C.OPEN}\n{history(turns)}\n{C.CLOSE}\n{ASK}"
    return [{"role": "system", "content": "You summarise."},
            {"role": "user", "content": [{"type": "text", "text": text}] if parts else text}]


def text_of(message):
    c = message["content"]
    return c if isinstance(c, str) else c[0]["text"]


class Fit(unittest.TestCase):
    CONTEXT = 6000

    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(Engine(tok, self.CONTEXT), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def prepare(self, messages, max_new=1000):
        return self.svc.prepare(messages, None, {}, max_new=max_new, guard=False)

    def test_a_compaction_too_long_fits_with_the_newest_kept(self):
        for parts in (False, True):
            messages = compact(200, parts)
            self.assertGreater(len(self.svc.render_ids(messages, None, {})), 3 * self.CONTEXT)
            ids, _, max_new = self.prepare(messages)
            self.assertEqual(max_new, 1000)
            self.assertLessEqual(len(ids) + max_new, self.CONTEXT)
            self.assertGreater(len(ids) + max_new, self.CONTEXT * 0.8)       # (not cut far more than needed)
            prompt = self.svc.tok.decode(ids)
            self.assertIn("left out here", prompt)
            self.assertIn("built " * 20 + "199", prompt)                    # the newest stays
            self.assertNotIn("step 0\n", prompt)                            # the oldest went
            self.assertIn(ASK, prompt)                                      # the instructions stay
            self.assertIn("Here is the conversation so far:", prompt)

    def test_shorten_cuts_at_a_line_and_says_how_much(self):
        messages = compact(10)
        at = C.quoted(messages)
        out = text_of(C.shorten(messages, at, 30)[-1])
        self.assertTrue(out.startswith(f"Here is the conversation so far:\n{C.OPEN}\n[... the oldest "))
        self.assertIn(" characters of the conversation are left out here", out)
        self.assertIn("...]\n[Tool result]: " + "built " * 20 + "0\n[User]: step 1", out)
        self.assertTrue(out.endswith(f"{C.CLOSE}\n{ASK}"))
        everything = text_of(C.shorten(messages, at, 10 ** 6)[-1])
        self.assertIn(f"left out here: the request did not fit the context ...]\n{C.CLOSE}\n{ASK}", everything)
        self.assertEqual(text_of(messages[-1]), compact(10)[-1]["content"])  # (a copy: the request is as it was)

    def test_what_fits_or_is_no_compaction_is_left_as_it_is(self):
        messages = compact(3)
        ids, _, _ = self.prepare(messages)
        self.assertEqual(ids, self.svc.render_ids(messages, None, {}))
        long = [{"role": "user", "content": history(200)}]               # no <conversation>: rejected as ever
        with self.assertRaises(ContextOverflow):
            self.prepare(long)
        later = compact(200) + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "go on"}]
        self.assertIsNone(C.quoted(later))                                # the quote must be the last message
        with self.assertRaises(ContextOverflow):
            self.prepare(later)


if __name__ == "__main__":
    unittest.main()
