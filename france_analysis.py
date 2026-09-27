"""What goes wrong on France (test only, no labels)? Uses D1's cached test probabilities.
    python france_analysis.py > artifacts/france_analysis.txt"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "code/business_entity_resolution/src")
import numpy as np

from er_pipeline import read_tsv, col, target_best

s1, s2, s3 = (read_tsv(f"dataset/test/test_source{i}.tsv") for i in (1, 2, 3))
n2 = s2.num_rows
C1 = {k: col(s1, k) for k in ("business_name", "business_address", "country")}
T = [{k: col(t, k) for k in ("business_name", "business_address")} for t in (s2, s3)]
ctry = np.array(C1["country"].to_pylist(), dtype=object)
Z = np.load("artifacts/D1/cache/test_scored.npz")
q, t, p = Z["q"], Z["t"], Z["prob"]
thr = json.loads(Path("artifacts/D1/run_report.json").read_text())["validation"]["threshold"]
fr = ctry[q] == "France"


def trec(x):
    s, r = (0, x) if x < n2 else (1, x - n2)
    return f"{T[s]['business_name'][int(r)].as_py()} | {T[s]['business_address'][int(r)].as_py()}"


def srec(x):
    return f"{C1['business_name'][int(x)].as_py()} | {C1['business_address'][int(x)].as_py()}"


rng = np.random.default_rng(0)
above = p >= thr
# contested French records: >= 2 French S1 above threshold
ta = t[above & fr]
cnt = np.bincount(ta, minlength=t.max() + 1)
contested = np.flatnonzero(cnt >= 2)
print(f"French above-threshold pairs {int((above & fr).sum()):,}; contested records {len(contested):,}")
sel = rng.choice(contested, min(15, len(contested)), replace=False)
for x in sel:
    m = (t == x) & above
    idx = np.flatnonzero(m)[np.argsort(-p[m])][:3]
    print(f"\n RECORD: {trec(x)}")
    for i in idx:
        print(f"   p={p[i]:.3f} S1: {srec(q[i])}")

# French S1 whose best candidate is below threshold (likely missed matches)
best = np.full(len(ctry), -1.0)
np.maximum.at(best, q, p)
frS = np.flatnonzero(ctry == "France")
low = frS[(best[frS] < thr) & (best[frS] > 0.2)]
print(f"\nFrench S1 with best p in (0.2, thr): {len(low):,} of {len(frS):,}")
for x in rng.choice(low, min(12, len(low)), replace=False):
    m = q == x
    idx = np.flatnonzero(m)[np.argsort(-p[m])][:2]
    print(f"\n S1: {srec(x)}")
    for i in idx:
        print(f"   p={p[i]:.3f} -> {trec(t[i])}")
