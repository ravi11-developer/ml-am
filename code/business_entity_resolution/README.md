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
`<DATA>/test/test_source{1,2,3}.tsv`. From `student_resource/`:

```bash
pip install -r code/business_entity_resolution/requirements.txt

# final submission (about 70 min on 16 threads / 16 GB RAM)
__FINAL_COMMAND__

python3 utils/validate_submission.py --matching output/matching_results.tsv     --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
Each run writes `matching_results.tsv` and `candidate_pairs.tsv` to `--output-dir`, and to
`--model-dir`: the LightGBM model, `run_report.json` (validation F0.5, blocking ceiling, config),
the threshold curve, feature importance, and `cache/` (validation and test probabilities).
Runs are deterministic for a fixed `--seed`.

Optional: per-entity expected-F0.5 selection from the cached probabilities (about 2 min, no
recomputation). It writes a new `matching_results.tsv` only if it beats the global threshold on
validation:
```bash
python code/business_entity_resolution/src/decide.py --run artifacts/<run> --out output/<run>_ef
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
| `--k-addr-pass` | 0 (off) | +~40% blocking time at 10–15 | Largest recall gain (0.84 → 0.955 at 10) |
| `--k-name-pass` | 0 (off) | small | No measurable gain on this data |
| `--indic-translit` | off | +~1 min preprocessing | Romanises Indic-script names/addresses |
| `--name-skeleton` | off | +~20 s | Phonetic-skeleton keys (transliteration / spelling variants) |
| `--name-clean` | off | ~0 | Leet digits → letters, drops web tokens (`com`, `www`) |
| `--cross-features` | off | +~5 min on test | Similarity to the entity's other top candidates (+0.002 F0.5) |
| `--rounds` / `--lr` | 1500 / 0.08 | Training and scoring scale with rounds | Early stopping on validation |
| `--skip-test` | off | Stops after validation | For quick model experiments |

To choose settings, read `val_f05` and `val_f05_blocking_ceiling` in `run_report.json`. If they
are close, the model is saturated and only better blocking (`--pool`, `--max-posting`) will
raise the score. On synthetic benchmarks, validation F0.5 matched true test F0.5 to within 0.003.

## Method summary
1. **Preprocessing** (pyarrow, vectorised). Indic-script text (Devanagari, Bengali, Gurmukhi,
   Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam) is romanised with a rule table built from
   Unicode character names (Python standard library only). Then NFKD with accents stripped,
   lowercase, `&`→`and`, punctuation removed, `560 001`→`560001`. Abbreviations (Rd→road,
   Pvt→private, …) are expanded on the vocabulary. In names, leet digits are mapped to letters
   and web tokens are dropped. A core name is built by dropping legal suffixes.
2. **Keys.** Word unigram + bigram TF-IDF for name and address, plus a phonetic skeleton per name
   token (sh/s, ph/f, x/ks, c/k, voiced/unvoiced merged, doubled letters collapsed, non-initial
   vowels dropped). The record vector is `√0.6·name ⊕ √0.4·address`.
3. **Blocking.** Per source there is a combined pass (top K by full cosine) plus an address-only
   pass (top `k_addr` by address cosine). Each pass retrieves a pool from an inverted index that
   skips keys found in more than `max_posting` target records, then re-ranks it by exact cosine.
   The union is the candidate set, and there is no Cartesian product. `candidate_pairs.tsv` is
   exactly what the model scores.
4. **Training data.** A sample of Source 1 entities is queried against the full S2/S3. Matches
   are labelled from the ground truth. Negatives are sampled at 1:5, 60% of them hard. K is
   chosen from the recall curve.
5. **Features.** Name/address word, bigram and skeleton Jaccard; TF-IDF cosines; Levenshtein and
   rapidfuzz ratios; core-name ratio; country / postal-code / house-number / first-token
   agreement; lengths; source flag; candidate context (rank, gap, ratio, count, rank in each
   pass); cross-candidate consistency (similarity to the entity's other top-5 candidates).
6. **Model.** LightGBM with early stopping on held-out entities. The threshold maximises exact
   macro F0.5 (singletons included), then the one-to-one rule is applied. The final model is
   refit on all sampled rows.
7. **Inference.** Test pairs are scored in batches. Every test S1 entity gets a row (empty for
   singletons), and the outputs are self-validated.

Country is used only as text and in an equality feature. It is never one-hot encoded or
filtered on, so France (present only in the test set) is handled exactly like US and India.

## Smoke test
```bash
python src/make_dummy_data.py --out dummy_dataset --n 20000
python src/run_pipeline.py --data-dir dummy_dataset --output-dir dummy_out --train-queries 20000
```
