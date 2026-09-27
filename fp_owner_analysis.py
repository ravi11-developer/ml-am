"""Are validation false positives records that belong to ANOTHER S1 entity (which at test time would
also be queried and win the one-to-one), or records that match nobody?
    python fp_owner_analysis.py artifacts/q30_s2"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

sys.path.insert(0, "code/business_entity_resolution/src")
from er_pipeline import target_best

run = Path(sys.argv[1])
rep = json.loads((run / "run_report.json").read_text())["validation"]
V = pd.read_parquet(run / "cache" / "val_scored.parquet")


def ids(path):
    return pacsv.read_csv(path, parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                          convert_options=pacsv.ConvertOptions(include_columns=["entity_id"],
                                                               column_types={"entity_id": pa.large_string()})
                          ).column(0).combine_chunks()


d = "dataset/train/"
s1 = pc.utf8_trim_whitespace(ids(d + "train_source1.tsv"))
tg = pc.utf8_trim_whitespace(pa.concat_arrays([ids(d + "train_source2.tsv"), ids(d + "train_source3.tsv")]))
g = pacsv.read_csv(d + "train_ground_truth.tsv", parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                   convert_options=pacsv.ConvertOptions(strings_can_be_null=False,
                                                        column_types={"source1_entity_id": pa.large_string(),
                                                                      "matched_entity_ids": pa.large_string()}))
L = pc.split_pattern(g.column("matched_entity_ids").combine_chunks(), ",")
par = pc.list_parent_indices(L)
gt_t = pc.index_in(pc.utf8_trim_whitespace(pc.list_flatten(L)), value_set=tg).to_numpy(zero_copy_only=False)
gt_q = pc.index_in(pc.utf8_trim_whitespace(g.column("source1_entity_id").combine_chunks().take(par)),
                   value_set=s1).to_numpy(zero_copy_only=False)
ok = ~(np.isnan(gt_t.astype(float)) | np.isnan(gt_q.astype(float)))
owner = np.full(len(tg), -1, np.int64)
owner[gt_t[ok].astype(np.int64)] = gt_q[ok].astype(np.int64)

p, y, t, q = V["prob"].to_numpy(), V["label"].to_numpy().astype(bool), V["t"].to_numpy(), V["q"].to_numpy()
tb = target_best(t, p)
keep = (p >= rep["threshold"]) & tb
fp = keep & ~y
own = owner[t[fp]]
print(f"threshold {rep['threshold']}, FP pairs {fp.sum():,}")
print(f"  FP target matches NO S1 at all (distractor / singleton record): {100 * (own < 0).mean():.1f}%")
print(f"  FP target belongs to ANOTHER S1 entity:                         {100 * (own >= 0).mean():.1f}%")
val_q = set(np.unique(q))
print(f"     ... and that owner is itself a validation query:             "
      f"{100 * np.mean([o in val_q for o in own[own >= 0]]):.1f}%  (only these can compete in validation)")
# all candidates: how often does the true owner of a candidate target exist but is not queried?
allown = owner[t]
print(f"  all val candidates whose target is owned by an S1 that is not queried in validation: "
      f"{100 * np.mean((allown >= 0) & (allown != q) & ~np.isin(allown, list(val_q))):.1f}%")
