"""Phase A6 + one-to-one activity on TEST (no labels): why does the leaderboard trail validation, and why
is D2 (LB 0.928) below D1 (LB 0.936)?   python test_gap_analysis.py > artifacts/test_gap_analysis.txt"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "code/business_entity_resolution/src")
import numpy as np
import pandas as pd

from er_pipeline import read_tsv, col, target_best

s1 = read_tsv("dataset/test/test_source1.tsv")
ctry = np.array(col(s1, "country").to_pylist(), dtype=object)
n1 = len(ctry)
res = {}
for run in ("D1", "D2"):
    Z = np.load(f"artifacts/{run}/cache/test_scored.npz")
    q, t, p = Z["q"], Z["t"], Z["prob"]
    thr = json.loads(Path(f"artifacts/{run}/run_report.json").read_text())["validation"]["threshold"]
    above = p >= thr
    tb = target_best(t, p)
    keep = above & tb
    res[run] = (q, t, p, keep, thr)
    print(f"\n### {run} (threshold {thr}): {len(q):,} candidates, {above.sum():,} above threshold, "
          f"{keep.sum():,} kept after one-to-one")
    print(f"  one-to-one removed {100 * (above & ~tb).sum() / max(above.sum(), 1):.2f}% of above-threshold pairs")
    # contested targets: >=2 S1 above threshold
    ta = t[above]
    cnt = np.bincount(ta)
    print(f"  targets claimed by >=2 S1 above threshold: {(cnt >= 2).sum():,} "
          f"({100 * (cnt >= 2).sum() / max((cnt >= 1).sum(), 1):.2f}% of claimed targets)")
    # close calls: loser prob within 0.05 of winner
    order = np.lexsort((-p[above], ta))
    ts, ps = ta[order], p[above][order]
    first = np.r_[True, ts[1:] != ts[:-1]]
    second = np.r_[False, ~first[1:]] & np.r_[True, first[:-1]]
    best = ps[first]
    sec = np.full(len(best), np.nan); sec[np.cumsum(first)[second] - 1] = ps[second]
    close = (best - sec) < 0.05
    print(f"  contested targets where the runner-up is within 0.05 of the winner: {int(np.nansum(close)):,}")
    has = np.bincount(q[keep], minlength=n1)
    for c in pd.unique(ctry):
        mk = ctry == c
        cq = mk[q]
        print(f"  {c:7s} S1 {mk.sum():>8,}  mean matches/S1 {has[mk].mean():.2f}  predicted singletons {100 * (has[mk] == 0).mean():5.1f}%"
              f"  mean p of kept {p[keep & cq].mean():.3f}  kept pairs with p<0.95: {100 * (p[keep & cq] < 0.95).mean():4.1f}%"
              f"  above-thr removed by 1-to-1: {100 * (above & ~tb & cq).sum() / max((above & cq).sum(), 1):.2f}%")

# D1 vs D2 disagreement
k1 = set(zip(res["D1"][0][res["D1"][3]].tolist(), res["D1"][1][res["D1"][3]].tolist()))
k2 = set(zip(res["D2"][0][res["D2"][3]].tolist(), res["D2"][1][res["D2"][3]].tolist()))
only1, only2 = k1 - k2, k2 - k1
print(f"\n### D1 vs D2 matches: common {len(k1 & k2):,}, only D1 {len(only1):,}, only D2 {len(only2):,}")
for name, s in (("only D1", only1), ("only D2", only2)):
    qs = np.array([a for a, _ in s], dtype=np.int64)
    print(f"  {name} by country: " + ", ".join(f"{c}={100 * (ctry[qs] == c).mean():.1f}%" for c in pd.unique(ctry)))
print("  (test S1 country mix: " + ", ".join(f"{c}={100 * (ctry == c).mean():.1f}%" for c in pd.unique(ctry)) + ")")

# raw examples of only-D2 matches (likely the source of the drop)
s2, s3 = read_tsv("dataset/test/test_source2.tsv"), read_tsv("dataset/test/test_source3.tsv")
n2 = s2.num_rows
cols = {k: (col(s1, "business_name"), col(s1, "business_address")) for k in ["s1"]}
T_name = [col(s2, "business_name"), col(s3, "business_name")]
T_addr = [col(s2, "business_address"), col(s3, "business_address")]
rng = np.random.default_rng(0)
for name, s in (("only D2 (D2 added)", only2), ("only D1 (D2 dropped)", only1)):
    lst = list(s)
    print(f"\n### 12 random pairs {name}")
    for i in rng.choice(len(lst), min(12, len(lst)), replace=False):
        qq, tt = lst[i]
        src, r = (0, tt) if tt < n2 else (1, tt - n2)
        print(f"  [{ctry[qq]}] {cols['s1'][0][qq].as_py()} | {cols['s1'][1][qq].as_py()}")
        print(f"        -> {T_name[src][r].as_py()} | {T_addr[src][r].as_py()}")
