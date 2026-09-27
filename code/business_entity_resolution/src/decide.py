"""Phase 3: per-entity expected-F0.5 selection from a run's cached probabilities (no recomputation).
    python code/business_entity_resolution/src/decide.py --run artifacts/D1 --out output/D1_ef [--test-dir dataset/test]
Tunes on the cached validation predictions, compares with the run's global threshold, and (if it wins)
rewrites matching_results.tsv for the cached test candidates. candidate_pairs.tsv is copied unchanged."""
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

from er_pipeline import macro_f05, target_best, read_tsv, col, log, write_lists


def select(q, p, nq, power, alpha, beta, gamma):
    """Per query: choose m maximising E[F0.5]; returns a keep mask (inputs already one-to-one filtered)."""
    pc_ = np.clip(p, 1e-6, 1 - 1e-6) ** power
    order = np.lexsort((-pc_, q))
    qs, ps = q[order], pc_[order]
    start = np.r_[True, qs[1:] != qs[:-1]]
    gidx = np.cumsum(start) - 1
    gstart = np.flatnonzero(start)
    cs = np.cumsum(ps)
    base = np.r_[0, cs][gstart][gidx]
    s_m = cs - base                                    # sum of top-m probs (m = rank)
    m = np.arange(len(qs)) - gstart[gidx] + 1
    tot = np.bincount(gidx, weights=ps)
    G = alpha * tot + beta
    ef = 1.25 * s_m / (m + 0.25 * G[gidx])
    log1m = np.bincount(gidx, weights=np.log1p(-np.minimum(ps, 1 - 1e-6)))
    ef0 = gamma * np.exp(log1m)                        # predict nothing
    best = np.maximum.reduceat(ef, gstart)
    best_m = np.zeros(len(gstart), np.int64)
    # smallest m reaching the group maximum
    hit = ef >= best[gidx] - 1e-12
    first_hit = np.full(len(gstart), np.iinfo(np.int64).max)
    np.minimum.at(first_hit, gidx[hit], m[hit])
    best_m = np.where(best > ef0, first_hit, 0)
    keep_sorted = m <= best_m[gidx]
    keep = np.zeros(len(q), bool)
    keep[order] = keep_sorted
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--no-test", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    rep = json.loads((run / "run_report.json").read_text())["validation"]
    V = pd.read_parquet(run / "cache" / "val_scored.parquet")
    vg = np.load(run / "cache" / "val_gt_cnt.npy")
    val_q, gt_cnt = vg[:, 0], vg[:, 1]
    qi = np.searchsorted(val_q, V["q"].to_numpy())
    p, y, t = V["prob"].to_numpy(), V["label"].to_numpy().astype(bool), V["t"].to_numpy()
    tb = target_best(t, p) if rep["one_to_one"] else np.ones(len(p), bool)
    nq = len(val_q)
    base = macro_f05(nq, qi[(p >= rep["threshold"]) & tb], y[(p >= rep["threshold"]) & tb], gt_cnt)
    log(f"global threshold {rep['threshold']}: val F0.5 {base:.4f}")
    qa, pa_, ya = qi[tb], p[tb], y[tb]
    best = (base, None)
    grid = itertools.product([1.0, 1.5, 2.0, 3.0, 4.0], [0.5, 0.65, 0.8, 1.0], [0.3, 0.6, 1.0, 1.5], [1.0, 2.0, 4.0, 8.0, 16.0])
    for prm in grid:
        k = select(qa, pa_, nq, *prm)
        f = macro_f05(nq, qa[k], ya[k], gt_cnt)
        if f > best[0]:
            best = (f, prm)
    log(f"best expected-F selection: val F0.5 {best[0]:.4f} params(power, alpha, beta, gamma)={best[1]}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "decision.json").write_text(json.dumps({"global_threshold_val_f05": base, "ef_val_f05": best[0],
                                                   "ef_params": best[1]}, indent=2))
    if a.no_test:
        return
    Z = np.load(run / "cache" / "test_scored.npz")
    q, t, prob = Z["q"], Z["t"], Z["prob"]
    ok = target_best(t, prob) if rep["one_to_one"] else np.ones(len(prob), bool)
    if best[1] is None:
        keep = ok & (prob >= rep["threshold"])
        log("expected-F did not beat the global threshold; using the threshold")
    else:
        idx = np.flatnonzero(ok)
        keep = np.zeros(len(prob), bool)
        keep[idx[select(q[idx], prob[idx], int(q.max()) + 1, *best[1])]] = True
    # ids only (no feature recomputation)
    ids = pa.concat_arrays([pc.utf8_trim_whitespace(col(read_tsv(Path(a.test_dir) / f"test_source{i}.tsv"), "entity_id"))
                            for i in (1, 2, 3)])
    n1 = read_tsv(Path(a.test_dir) / "test_source1.tsv").num_rows

    class _D:  # minimal Split stand-in for write_lists
        pass
    D = _D(); D.ids, D.n1 = ids, n1
    shutil.copyfile(Path(a.run.replace("artifacts", "output")) / "candidate_pairs.tsv", out / "candidate_pairs.tsv") \
        if (Path(a.run.replace("artifacts", "output")) / "candidate_pairs.tsv").exists() else None
    write_lists(out / "matching_results.tsv", D, q[keep], t[keep], prob[keep], "matched_entity_ids")
    has = np.bincount(q[keep], minlength=n1) > 0
    log(f"wrote {out / 'matching_results.tsv'}: {int(keep.sum()):,} matches, {int((~has).sum()):,} predicted singletons")


if __name__ == "__main__":
    main()
