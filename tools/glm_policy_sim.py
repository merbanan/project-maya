#!/usr/bin/env python3
"""VRAM expert-cache policies replayed over a STRATA_GLM_ROUTE_LOG file ("layer e0 .. ek fetch miss cpu | near..."):
LRU, aged LFU (counts halved every 4096 routes, recency breaking ties - the engine's default), and a profile-pinned
set (the most-routed experts of the first half of the log) with LRU for the rest.  Hit rate over the second half.

    python glm_policy_sim.py LOG SLOTS [PIN_SHARE]
"""
import collections
import sys


def routes(path):
    out = []
    for line in open(path):
        head = line.split("|")[0].split()
        if len(head) < 5:
            continue
        layer = int(head[0])
        ids = [int(x) for x in head[1:-3]]
        out.append((layer, ids))
    return out


def run(rt, slots, policy, pinned=frozenset()):
    cache = collections.OrderedDict()            # key -> [count, tick]
    tick = 0
    hits = total = 0
    half = len(rt) // 2
    cap = slots - len(pinned)
    for n, (layer, ids) in enumerate(rt):
        if policy == "lfu" and n and n % 4096 == 0:
            for v in cache.values():
                v[0] >>= 1
        for e in ids:
            k = (layer, e)
            tick += 1
            hit = k in pinned or k in cache
            if n >= half:
                hits += hit
                total += 1
            if k in pinned:
                continue
            if k in cache:
                cache[k][0] += 1
                cache[k][1] = tick
                cache.move_to_end(k)
                continue
            if len(cache) >= cap:
                if policy == "lfu":
                    victim = min(cache.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
                else:
                    victim = next(iter(cache))
                del cache[victim]
            cache[k] = [1, tick]
    return hits / max(total, 1)


def main():
    path, slots = sys.argv[1], int(sys.argv[2])
    pin_share = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    rt = routes(path)
    print("%d routes, %d slots" % (len(rt), slots))
    print("LRU        %.3f" % run(rt, slots, "lru"))
    print("aged LFU   %.3f" % run(rt, slots, "lfu"))
    cnt = collections.Counter((l, e) for l, ids in rt[: len(rt) // 2] for e in ids)
    for share in sorted({pin_share, 0.25, 0.75}):
        pinned = frozenset(k for k, _ in cnt.most_common(int(slots * share)))
        print("pin %.2f + LRU %.3f" % (share, run(rt, slots, "lru", pinned)))


if __name__ == "__main__":
    main()
