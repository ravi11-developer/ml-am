"""Tune the final decision rule on the TEST-LIKE validation (all S1 compete, see testlike_eval.py) and
apply it to a run's cached test probabilities. No retraining.
    python code/business_entity_resolution/src/decide_testlike.py --run D1 --out output/D1_tl
Rules searched (all after global one-to-one):
  thr     probability threshold
  margin  drop a record entirely if the runner-up S1's probability is within `margin` of the winner's
          (ambiguous same-name siblings; F0.5 prefers dropping over guessing)
  cap     keep at most `cap` records per S1 per source (ground truth has <= 5-6)
"""
import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from er_pipeline import macro_f05, read_tsv, col, log, write_lists


def contest_stats(t, p):
    """Per row: winner flag (best S1 for its record) and the record's runner-up probability."""
    order = np.lexsort((-p, t))
    ts, ps = t[order], p[order]
    first = np.r_[True, ts[1:] != ts[:-1]]
    gid = np.cumsum(first) - 1
    gstart = np.flatnonzero(first)
    size = np.diff(np.r_[gstart, len(ts)])
    second = np.where(size > 1, ps[np.minimum(gstart + 1, len(ps) - 1)], 0.0)
    win = np.zeros(len(t), bool); win[order[first]] = True
    run2 = np.empty(len(t), np.float32); run2[order] = second[gid]
    return win, run2


def per_source_rank(q, src, p):
    order = np.lexsort((-p, q * 2 + src))
    k = (q * 2 + src)[order]
    start = np.r_[True, k[1:] != k[:-1]]
    gs = np.maximum.accumulate(np.where(start, np.arange(len(k)), 0))
    r = np.empty(len(q), np.int32); r[order] = np.arange(len(k)) - gs + 1
    return r


def decide(q, t, src, p, thr, margin, cap, win=None, run2=None):
    if win is None:
        win, run2 = contest_stats(t, p)
    keep = win & (p >= thr) & ~((run2 >= thr) & (p - run2 < margin))
    if cap < 99:
        idx = np.flatnonzero(keep)
        r = per_source_rank(q[idx].astype(np.int64), src[idx].astype(np.int64), p[idx])
        keep[idx[r > cap]] = False
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="D1 / D2 / ...: uses artifacts/testlike/<run>_pairs.parquet and artifacts/<run>/cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--no-test", action="store_true")
    a = ap.parse_args()
    own = Path(f"artifacts/{a.run}/cache/testlike_pairs.parquet")      # written by the pipeline (--target-features)
    if own.exists():
        P = pd.read_parquet(own)
        Eg = np.load(f"artifacts/{a.run}/cache/testlike_E_gtcnt.npy")
    else:                                                              # written by testlike_eval.py
        P = pd.read_parquet(f"artifacts/testlike/{a.run}_pairs.parquet")
        Eg = np.load("artifacts/testlike/eval_E_gtcnt.npy")
    E, gt_cnt = Eg[:, 0], Eg[:, 1]
    q, t, src, p, y = (P[c].to_numpy() for c in ("q", "t", "src", "prob", "label"))
    y = y.astype(bool)
    inE = np.isin(q, E)
    qi = np.searchsorted(E, q)
    win, run2 = contest_stats(t, p)
    rep = json.loads(Path(f"artifacts/{a.run}/run_report.json").read_text())["validation"]

    def f(keep):
        k = keep & inE
        return macro_f05(len(E), qi[k], y[k], gt_cnt)

    base = f(decide(q, t, src, p, rep["threshold"], 0.0, 99, win, run2))
    log(f"test-like F0.5 with the run's own rule (thr {rep['threshold']}): {base:.4f}")
    best = (base, (rep["threshold"], 0.0, 99))
    for thr, margin, cap in itertools.product([round(x, 2) for x in np.arange(0.3, 0.97, 0.03)],
                                              [0.0, 0.02, 0.05, 0.1, 0.2], [99, 5, 4, 3]):
        v = f(decide(q, t, src, p, thr, margin, cap, win, run2))
        if v > best[0]:
            best = (v, (thr, margin, cap))
    log(f"best test-like F0.5 {best[0]:.4f} with (thr, margin, cap) = {best[1]}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "decision_testlike.json").write_text(json.dumps({"run_rule_f05": base, "best_f05": best[0],
                                                            "thr_margin_cap": best[1]}, indent=2))
    if a.no_test:
        return
    Z = np.load(f"artifacts/{a.run}/cache/test_scored.npz")
    tq, tt, ts_, tp = Z["q"], Z["t"], Z["src"], Z["prob"]
    keep = decide(tq, tt, ts_, tp, *best[1])
    ids = pa.concat_arrays([pc.utf8_trim_whitespace(col(read_tsv(Path(a.test_dir) / f"test_source{i}.tsv"), "entity_id"))
                            for i in (1, 2, 3)])
    n1 = read_tsv(Path(a.test_dir) / "test_source1.tsv").num_rows

    class _D:
        pass
    Dm = _D(); Dm.ids, Dm.n1 = ids, n1
    write_lists(out / "matching_results.tsv", Dm, tq[keep], tt[keep], tp[keep], "matched_entity_ids")
    shutil.copyfile(f"output/{a.run}/candidate_pairs.tsv", out / "candidate_pairs.tsv")
    has = np.bincount(tq[keep], minlength=n1) > 0
    log(f"wrote {out}: {int(keep.sum()):,} matches, {int((~has).sum()):,} predicted singletons")


if __name__ == "__main__":
    main()
