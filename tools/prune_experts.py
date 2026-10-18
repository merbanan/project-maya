"""Domain prune lists from routing traces (--dump-routing, e.g. tools/mimo_calib.py): each layer's experts scored by
the router weight they were given over the traces (REAP's gate term; the expert output norm is not in the trace),
the lowest-scored fraction of each layer listed for STRATA_GLM_PRUNE ("layer e e e ..." lines).

    python tools/prune_experts.py --traces '/tmp/calib/*.routes' --n-expert 256 --ratios 0.05,0.1,0.2,0.25 \
        --out /tmp/prune    ->  /tmp/prune05.txt /tmp/prune10.txt ...
"""
import argparse
import glob

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--traces", required=True)
ap.add_argument("--n-expert", type=int, default=256)
ap.add_argument("--ratios", default="0.05,0.1,0.2,0.25")
ap.add_argument("--out", required=True, help="a path prefix: <out>NN.txt per ratio")
a = ap.parse_args()

NE = a.n_expert
w = {}   # layer -> weight sums
c = {}   # layer -> route counts
n_rec = 0
for f in sorted(glob.glob(a.traces)):
    b = np.fromfile(f, dtype=np.int32)
    i = 0
    while i + 2 <= len(b):
        il, k = int(b[i]), int(b[i + 1])
        ids = b[i + 2:i + 2 + k]
        ws = b[i + 2 + k:i + 2 + 2 * k].view(np.float32)
        i += 2 + 2 * k
        if il not in w:
            w[il] = np.zeros(NE)
            c[il] = np.zeros(NE)
        np.add.at(w[il], ids, ws)
        np.add.at(c[il], ids, 1)
        n_rec += 1
print("%d routes over %d layers" % (n_rec, len(w)))
for r in (float(x) for x in a.ratios.split(",")):
    n = int(round(r * NE))
    lost, never = [], 0
    with open("%s%02d.txt" % (a.out, round(r * 100)), "w") as o:
        o.write("# prune %.0f%%: the %d lowest router-weight experts of each layer (%d routes)\n" % (r * 100, n, n_rec))
        for il in sorted(w):
            order = np.argsort(w[il], kind="stable")[:n]
            lost.append(w[il][order].sum() / max(1e-12, w[il].sum()))
            never += int((c[il][order] == 0).sum())
            o.write("%d %s\n" % (il, " ".join(str(int(e)) for e in sorted(order))))
    print("prune %4.0f%%: %3d experts a layer; router weight removed: mean %.3f%%, max %.3f%% (layer); %d never routed" % (
        r * 100, n, 100 * np.mean(lost), 100 * np.max(lost), never))
