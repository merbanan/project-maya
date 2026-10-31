"""tools/nh/compare.py - maya-nh's logits (--dump-logits) against tools/nh/ref.py's (--dump), or two engine dumps.

    python tools/nh/compare.py engine.bin ref.npz [more.bin ...]

Per row: the top token of each, the KL(ref || engine) of the softmaxes, the largest logit difference and the rank of
the reference's top token in the engine's logits.  The first file is the one compared against the second (the
reference); more files are compared against the reference too.
"""
from __future__ import annotations

import struct
import sys

import numpy as np


def load(path: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (ids, logits [rows][vocab]); an engine dump (NHL1) or ref.py's npz."""
    if path.endswith(".npz"):
        z = np.load(path)
        return z["ids"], z["logits"].astype(np.float64)
    with open(path, "rb") as f:
        if f.read(4) != b"NHL1":
            raise SystemExit("%s: not a maya-nh logits dump" % path)
        rows, vocab, n = struct.unpack("<3i", f.read(12))
        ids = np.frombuffer(f.read(4 * n), dtype=np.int32)
        lg = np.frombuffer(f.read(4 * rows * vocab), dtype=np.float32).reshape(rows, vocab)
    return ids, lg.astype(np.float64)


def logsoftmax(x: np.ndarray) -> np.ndarray:
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    rid, ref = load(sys.argv[2])
    worst = 0.0
    for path in [sys.argv[1]] + sys.argv[3:]:
        eid, eng = load(path)
        if not np.array_equal(eid, rid):
            print("%s: other ids than the reference's" % path)
            return 1
        n = min(len(eng), len(ref))
        a, b = eng[-n:], ref[-n:]
        la, lb = logsoftmax(a), logsoftmax(b)
        kl = (np.exp(lb) * (lb - la)).sum(-1)
        print("%s against %s (%d rows):" % (path, sys.argv[2], n))
        for r in range(n):
            ta, tb = int(a[r].argmax()), int(b[r].argmax())
            rank = int((a[r] > a[r][tb]).sum())
            print("  row %3d: top %6d / %6d  KL %.5f  max |d| %.4f  ref top's rank %d" %
                  (r, ta, tb, kl[r], np.abs(a[r] - b[r]).max(), rank))
        print("  mean KL %.5f, max %.5f, top agrees %d / %d" % (kl.mean(), kl.max(),
                                                               int((a.argmax(-1) == b.argmax(-1)).sum()), n))
        worst = max(worst, float(kl.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
