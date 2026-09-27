# Business Entity Resolution — Amazon ML Challenge 2026 (large-scale pipeline)

Built for the full data (~1.5–2M Source 1 rows, ~4–6M rows each in Source 2/3).
Uses only the provided TSVs: no external data, APIs or geocoding. The final model is LightGBM
(MIT licence), far below the 8B-parameter limit.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt
```
Plan for about 16 GB RAM. `sparse_dot_topn` makes blocking multithreaded; without it the pipeline
falls back to scipy, which produces identical candidates but is single-threaded and slower.

## Reproduce (data → blocking → matching → output)
Data layout: `<DATA>/train/train_source{1,2,3}.tsv`, `<DATA>/train/train_ground_truth.tsv`,
`<DATA>/test/test_source{1,2,3}.tsv`.

```bash
# 1) FAST preset: get a valid submission quickly
python src/run_pipeline.py --data-dir <DATA> --output-dir output_fast \
    --train-queries 30000 --pool 40 --k-max 10 --fast-features

# 2) DEFAULT preset: recommended
python src/run_pipeline.py --data-dir <DATA> --output-dir output

# 3) HIGH-RECALL preset: if time allows
python src/run_pipeline.py --data-dir <DATA> --output-dir output_hr --pool 150 --max-posting 6000
```
Each run writes `matching_results.tsv` and `candidate_pairs.tsv` to the output directory, and
`models/` (the LightGBM model, `run_report.json` with validation F0.5 and the blocking ceiling,
the threshold curve and feature importance). Runs are deterministic for a fixed `--seed`.

The notebook `notebooks/business_entity_resolution.ipynb` runs the same code step by step and adds
a threshold plot and a false-positive / missed-match inspection. Set `DATA_DIR` in its config cell.

Validate before uploading (from `student_resource/`):
```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Knobs: time vs accuracy
| Flag | Default | Time effect | Accuracy effect |
|---|---|---|---|
| `--train-queries` | 200000 | Training side scales linearly | Small gains past ~100k; cheapest place to cut |
| `--pool` | 100 | Blocking scales roughly linearly | Biggest recall lever |
| `--max-posting` | 3000 | Cost grows fast when raised | More recall |
| `--k` / `--k-max` | auto / 20 | Test feature time scales with K | Auto-K keeps 99.5% of reachable recall |
| `--fast-features` | off | Saves ~30% of feature time | Small loss |
| `--threshold` | tuned | — | Tuned value is within ~0.001 of 0.75 / 0.80 |
| `--one-to-one` | auto | — | Each S2/S3 record goes to one S1 only (enabled if the ground truth confirms it) |

To choose settings, read `val_f05` and `val_f05_blocking_ceiling` in `run_report.json`. If they
are close, the model is saturated and only better blocking (`--pool`, `--max-posting`) will
raise the score. On synthetic benchmarks, validation F0.5 matched true test F0.5 to within 0.003.

## Method summary
1. **Preprocessing** (pyarrow, vectorised). NFKD normalisation with accents stripped, lowercase,
   `&`→`and`, punctuation removed, `560 001`→`560001`. Abbreviations (Rd→road, Pvt→private, …)
   are expanded on the vocabulary. A core name is built by dropping legal suffixes.
2. **Blocking.** Each record gets word unigram + bigram TF-IDF vectors for name and address,
   combined as 0.6·name + 0.4·address. Stage 1 retrieves a pool per source from an inverted
   index that skips tokens found in more than `max_posting` target records. Stage 2 re-ranks
   that pool by full cosine and keeps the top K per source. There is no Cartesian product.
   The candidate set written to `candidate_pairs.tsv` is exactly what the model scores.
3. **Training data.** A sample of Source 1 entities is queried against the full S2/S3. Matches
   are labelled from the ground truth. Negatives are sampled at 1:5, 60% of them hard (from
   each entity's top 3 candidates). K is chosen from the recall curve.
4. **Features (29).** Name/address word Jaccard (unigram and bigram), name Levenshtein ratio,
   TF-IDF cosines, rapidfuzz token-sort / token-set / partial ratios, core-name ratio,
   country / postal-code / house-number / first-token agreement, lengths, source flag, and
   candidate-context features (rank, gap and ratio to the best candidate, candidate count).
5. **Model.** LightGBM with early stopping on held-out entities. The threshold maximises exact
   macro F0.5 (singletons included), then the one-to-one rule is applied. The final model is
   refit on all sampled rows.
6. **Inference.** Test pairs are scored in batches. Every test S1 entity gets a row (empty for
   singletons), and the outputs are self-validated.

Country is used only as text and in an equality feature. It is never one-hot encoded or
filtered on, so France (present only in the test set) is handled exactly like US and India.

## Smoke test
```bash
python src/make_dummy_data.py --out dummy_dataset --n 20000
python src/run_pipeline.py --data-dir dummy_dataset --output-dir dummy_out --train-queries 20000
```
