"""serve/compaction.py - a client's request to compact a conversation, when it is too long for the context: the oldest
part of the conversation it quotes is left out until the request fits, so a conversation can always be compacted.

opencode compacts a conversation that fills the context by sending it back as text - one user message with the
conversation between <conversation> tags and what to make of it - and asking for a summary. A conversation that is
already near the context does not fit that request with room for the summary, and the server's 400
(context_length_exceeded) is the very thing that made the client compact: it has nothing left to try ("Session too
large to compact - context exceeds model limit even after stripping media"), and the session is stuck. Every other
request is still rejected, never truncated.
"""
from __future__ import annotations

OPEN, CLOSE = "<conversation>", "</conversation>"
GAP = "[... the oldest {n:,} characters of the conversation are left out here: the request did not fit the context ...]"
ROOM = 8192                 # the room for the summary when the request leaves it open (max tokens unset)


def _texts(message):
    """-> [(key, text)]: a message's text, as a string (key None) or as the text parts of a list (their index)."""
    c = message.get("content")
    if isinstance(c, str):
        return [(None, c)]
    if isinstance(c, list):
        return [(k, p.get("text") or "") for k, p in enumerate(c) if isinstance(p, dict) and p.get("type") == "text"]
    return []


def _span(text):
    """-> (start, end) of the conversation between the tags, or None."""
    a = text.find(OPEN)
    b = text.rfind(CLOSE)
    return (a + len(OPEN), b) if a >= 0 and b > a else None


def quoted(messages):
    """-> (message index, part key) of the conversation a compaction request quotes: the last message, a user's, with
    the conversation between <conversation> tags - or None for any other request."""
    if not messages or messages[-1].get("role") != "user":
        return None
    for key, text in _texts(messages[-1]):
        if _span(text):
            return len(messages) - 1, key
    return None


def size(messages, at):
    """The quoted conversation's length in characters."""
    i, key = at
    text = dict(_texts(messages[i]))[key]
    a, b = _span(text)
    return b - a


def shorten(messages, at, chars):
    """-> a copy of the messages with the oldest `chars` characters of the quoted conversation (to the end of that
    line) left out and said so; the instructions around it and the newest of the conversation stay."""
    i, key = at
    text = dict(_texts(messages[i]))[key]
    a, b = _span(text)
    cut = text.find("\n", a + chars, b)
    cut = b if cut < 0 else cut
    text = text[:a] + "\n" + GAP.format(n=cut - a) + ("\n" if cut == b else "") + text[cut:]
    m = dict(messages[i])
    if key is None:
        m["content"] = text
    else:
        m["content"] = [dict(p, text=text) if k == key else p for k, p in enumerate(m["content"])]
    return messages[:i] + [m] + messages[i + 1:]
