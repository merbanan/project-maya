"""A hybrid pack: one quant's pack (the primary: its dense weights, router and most expert layers) with chosen MoE
layers' routed experts taken from a second quant of the same model (the secondary) - e.g. MiMo BPW2.5 with ten
layers' experts from BPW2.0, ~4 GiB smaller so that every expert fits VRAM + RAM on a 62 GB box.

    python tools/hybrid_pack.py --primary M25/pack --secondary M20/pack --layers 37-46 --out M25/pack-h37

The output sits beside the primary pack (the engine finds shards next to the pack directory): the primary pack's
files are linked in, native_experts.txt is rewritten with the secondary's rows for those layers (each row naming its
shard) and a "# extra shards:" header line; the secondary's shards are linked next to the primary's.
"""
import argparse
import os
import re


def parse_layers(spec):
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def rows_of(path):
    head, rows = [], {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("#"):
            head.append(line)
        elif line.strip():
            rows[int(line.split()[0])] = line
    return head, rows


def first_shard(head):
    for h in head:
        m = re.search(r"absolute offsets in ([^,\s)]+)", h)
        if m:
            return m.group(1)
    raise SystemExit("no 'absolute offsets in' header")


ap = argparse.ArgumentParser()
ap.add_argument("--primary", required=True)
ap.add_argument("--secondary", required=True)
ap.add_argument("--layers", required=True, help="MoE layers whose experts come from the secondary, e.g. 37-46")
ap.add_argument("--out", required=True)
a = ap.parse_args()

prim, sec, out = (os.path.abspath(p) for p in (a.primary, a.secondary, a.out))
if os.path.dirname(out) != os.path.dirname(prim):
    raise SystemExit("--out must sit beside the primary pack (the engine finds the shards next to the pack)")
layers = parse_layers(a.layers)
ph, pr = rows_of(prim + "/native_experts.txt")
sh, sr = rows_of(sec + "/native_experts.txt")
s1 = first_shard(sh)
os.makedirs(out, exist_ok=True)
for f in os.listdir(prim):
    if f == "native_experts.txt":
        continue
    dst = os.path.join(out, f)
    if not os.path.lexists(dst):
        os.symlink(os.path.join(prim, f), dst)
# the secondary's shards beside the primary's (the -NNNNN-of-MMMMM siblings of its shard 1)
m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", s1)
names = ["%s-%05d-of-%s.gguf" % (m.group(1), i, m.group(3)) for i in range(1, int(m.group(3)) + 1)] if m else [s1]
sdir, pdir = os.path.dirname(sec), os.path.dirname(prim)
for n in names:
    dst = os.path.join(pdir, n)
    if not os.path.lexists(dst):
        os.symlink(os.path.join(sdir, n), dst)
saved, lines = 0, []
for il in sorted(pr):
    if il in layers:
        if il not in sr:
            raise SystemExit("layer %d is not in the secondary" % il)
        f = sr[il].split()
        if len(f) == 8:          # (a v3 row without a shard: the secondary's shard 1)
            f.append(s1)
        saved += (int(pr[il].split()[4]) - int(f[4])) * 256
        lines.append(" ".join(f))
    else:
        lines.append(pr[il])
with open(out + "/native_experts.txt", "w", encoding="utf-8") as o:
    for h in ph:
        o.write(h + "\n")
    o.write("# extra shards: %s\n" % s1)
    o.write("# hybrid: layers %s from %s\n" % (a.layers, sec))
    for l in lines:
        o.write(l + "\n")
print("hybrid pack %s: %d layers from the secondary, %.2f GiB smaller" % (out, len(layers), saved / 2**30))
