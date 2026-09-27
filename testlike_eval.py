"""Phase A1: TEST-LIKE validation. Blocks ALL train S1 (as on test), scores 100k held-out S1 entities
(never used to train D1 or D2) plus every S1 competing for the same S2/S3 records, then applies the
one-to-one rule across all of them, exactly like the test set.
    python testlike_eval.py > artifacts/testlike_eval.txt
Saves scored pairs to artifacts/testlike/{D1,D2}_pairs.parquet for decision-rule tuning."""
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, "code/business_entity_resolution/src")
import lightgbm as lgb
import numpy as np
import pandas as pd

import er_pipeline as ep
from er_pipeline import (Config, prepare_split, generate_candidates, add_group_features, compute_anchors,
                         pair_features, label_pairs, target_best, macro_f05, log)

N_EVAL = 100_000
cfg = Config(data_dir=sys.argv[1] if len(sys.argv) > 1 else "dataset", max_posting=10000, rerank_pool=150, k_addr_pass=15,
             indic_translit=True, name_skeleton=True, name_clean=True, cross_features=True)
D = prepare_split(cfg, "train")
C = generate_candidates(D, np.arange(D.n1, dtype=np.int32), 19, cfg)
C = C.drop(columns=["pass_rank_name", "pass_rank_rev"], errors="ignore")
C["label"] = label_pairs(D, C["q"].to_numpy(), C["t"].to_numpy())
gc.collect()

# held-out entities: not in D1's (100k) or D2's (250k) training sample (both drawn with rng(42))
used = np.zeros(D.n1, bool)
for nq in (100_000, 250_000):
    used[np.random.default_rng(42).choice(D.n1, min(nq, D.n1), replace=False)] = True
if not (~used).any():          # tiny smoke-test data: evaluate on everything
    used[:] = False
free = np.flatnonzero(~used)
E = np.sort(np.random.default_rng(7).choice(free, min(N_EVAL, len(free)), replace=False))
isE = np.zeros(D.n1, bool); isE[E] = True
gt_cnt = np.bincount(np.searchsorted(E, D.gt_q[isE[D.gt_q]]), minlength=len(E))
TE = np.zeros(D.nT, bool); TE[C["t"].to_numpy()[isE[C["q"].to_numpy()]]] = True
log(f"eval entities {len(E):,}; their candidate records {TE.sum():,}")

Path("artifacts/testlike").mkdir(parents=True, exist_ok=True)
np.save("artifacts/testlike/eval_E_gtcnt.npy", np.c_[E, gt_cnt])
runs = {"D2": dict(k_addr=15, cross=True), "D1": dict(k_addr=10, cross=False)}   # D2 first: no copy of C
C["pass_rank_name"] = np.nan
C["pass_rank_rev"] = np.nan
for run, spec in runs.items():
    rep = json.loads(Path(f"artifacts/{run}/run_report.json").read_text())["validation"]
    booster = lgb.Booster(model_file=f"artifacts/{run}/lgbm_final.txt")
    names = booster.feature_name()
    if spec["k_addr"] < 15:   # shrink C in place to this run's candidate set, recompute per-S1 group features
        C = C[((C["pass_rank_all"] <= 19) | (C["pass_rank_addr"] <= spec["k_addr"])).to_numpy()]
        C = add_group_features(C.drop(columns=["rank_in_source", "rank_overall", "gap_to_best", "ratio_to_best",
                                               "n_cand_s1"]).reset_index(drop=True))
        C.loc[C["pass_rank_addr"] > spec["k_addr"], "pass_rank_addr"] = np.nan
        gc.collect()
    rcfg = Config(**{**cfg.__dict__, "cross_features": spec["cross"], "k_addr_pass": spec["k_addr"]})
    if spec["cross"]:
        D.anchors = compute_anchors(C, D.n1, rcfg.n_anchors)
    P = C[TE[C["t"].to_numpy()]].reset_index(drop=True)
    gc.collect()
    log(f"[{run}] scoring {len(P):,} pairs (eval entities + all competitors)")
    prob = np.empty(len(P), np.float32)
    for s in range(0, len(P), cfg.feature_batch):
        X = pair_features(D, P.iloc[s:s + cfg.feature_batch], rcfg)
        prob[s:s + cfg.feature_batch] = booster.predict(X[names].to_numpy(np.float32), num_threads=cfg.n_threads)
    P["prob"] = prob
    P[["q", "t", "src", "block_score", "prob", "label"]].to_parquet(f"artifacts/testlike/{run}_pairs.parquet")

    q, t, y = P["q"].to_numpy(), P["t"].to_numpy(), P["label"].to_numpy().astype(bool)
    inE = isE[q]
    qi = np.searchsorted(E, q)
    tb_all = target_best(t, prob)                           # test-like: every S1 competes
    tb_E = np.zeros(len(P), bool); tb_E[np.flatnonzero(inE)[target_best(t[inE], prob[inE])]] = True  # old val-style

    def f(thr, tb):
        k = inE & (prob >= thr) & tb
        return macro_f05(len(E), qi[k], y[k], gt_cnt)

    thr = rep["threshold"]
    print(f"\n### {run} (trained threshold {thr}; old validation {rep['val_f05']:.4f})")
    print(f"  TEST-LIKE (all S1 compete, one-to-one): {f(thr, tb_all):.4f}")
    print(f"  val-style (only eval S1 compete):        {f(thr, tb_E):.4f}")
    print(f"  no one-to-one:                           {f(thr, np.ones(len(P), bool)):.4f}")
    print(f"  blocking ceiling:                        {macro_f05(len(E), qi[inE & y], y[inE & y], gt_cnt):.4f}")
    grid = [round(x, 2) for x in np.arange(0.30, 0.99, 0.02)]
    curve = [(x, f(x, tb_all)) for x in grid]
    bx, bf = max(curve, key=lambda z: z[1])
    print(f"  test-like threshold curve: best {bf:.4f} at {bx}; " +
          " ".join(f"{x}:{v:.4f}" for x, v in curve[::3]))
    del P, prob, X
    gc.collect()
log("done")
