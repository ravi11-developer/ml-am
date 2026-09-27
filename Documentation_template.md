# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
A blocking + classifier pipeline: sparse TF-IDF retrieval over word, bigram and phonetic-skeleton
keys, with a second address-only retrieval pass, followed by a LightGBM pair classifier and exact
macro-F0.5 decision tuning with a one-to-one constraint. The key gains came from (1) fixing a
row-alignment bug in our token pipeline, (2) the address-only pass, which recovers records whose
names were rewritten (native-script, domain-style or invented trade names), and (3) romanising
Indic-script text with a rule-based, stdlib-only transliterator. Validation macro F0.5 rose from
0.72 to __VAL_F05__.

---

## 2. Methodology

### 2.1 Problem Analysis
We compared true pairs that blocking missed with true pairs it found (3,000 training S1 entities,
10,445 true pairs). After the first fixes, the misses were dominated by:

| Noise pattern (target record) | Share of missed pairs | Example |
|---|---|---|
| Name written in an Indic script (Devanagari, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati) | 42% | `लक्ष्मी फूड्स प्राइवेट लिमिटेड` = Laxmi Foods Private Limited |
| Name replaced by a domain, handle or concatenated token | 22% | `krishnafoods.com`, `#danieljoaquin`, `aconsu1tants.com` |
| Typos, dropped / swapped / duplicated tokens, invented names at the same address | 22% | `Mienstils`, `Delta Delta Crsent LLC`, `Keloonyx` |
| Target address empty | 9% | – |
| Leet digits | 3% | `B1ack`, `inn0vative`, `6lobal` |

Other observations: 55% of the missed pairs still had address cosine ≥ 0.5, and 35% were crowded
out of the top K by *other* businesses with an identical name. Country labels never disagree
within a true pair. Addresses show field reordering, abbreviation vs full state names
(`OH`/`Ohio`, `TG`/`తెలంగాణ`), truncated or zero-padded house numbers (`1344`→`344`,
`709`→`00709`) and injected prefixes (`H.no`, `Door No`, `#`). Ground truth is strictly one-to-one
(no S2/S3 record belongs to two S1 entities) and 5.6% of S1 entities are singletons.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier  
**Core Innovation:** Multi-pass sparse blocking (combined name+address pass ∪ address-only pass)
over word, bigram and **phonetic-skeleton** keys, with **Indic-script romanisation** derived only
from Unicode character names (Python standard library). The classifier also gets
**cross-candidate consistency features**: S2/S3 copies of a business agree with each other even
when the S1 record is noisy.

---

## 3. Candidate Generation (Blocking)

- **Text preparation:** Indic-script characters are romanised with a `str.translate` table built
  from `unicodedata` names (consonant + inherent vowel, vowel signs, virama). Then: NFKD, accents
  stripped, lowercase, `&`→`and`, punctuation to spaces, Indian PIN `560 001`→`560001`.
  Abbreviations (`rd`→road, `pvt`→private, …) are expanded on the vocabulary. Name tokens with
  leet digits are mapped back to letters, and web tokens (`com`, `www`) are dropped.
- **Blocking keys used:** word unigrams + bigrams for name and address, plus a phonetic
  skeleton per name token (merges sh/s, ph/f, x/ks, c/k, voiced/unvoiced consonants, collapses
  doubled letters, drops non-initial vowels). The skeleton maps `Laxmi`, `Lakshmi` and the
  romanised `लक्ष्मी` to the same key (`lksm`), and `Private` / `प्राइवेट` / `பிரைவேட்` to `prvt`.
  All keys are IDF-weighted, and each record vector is `√0.6·name ⊕ √0.4·address`, L2-normalised
  per field.
- **Passes:** per source (S2, S3) and per S1 query: (1) the combined-cosine pass keeps the top K=19
  (K chosen from the recall curve); (2) an address-only pass keeps the top __KADDR__ by address
  cosine. Each pass retrieves a pool of 100–150 from an inverted index that skips keys found in
  more than 10,000 target records (`sparse_dot_topn`, multithreaded), then re-ranks the pool by
  exact cosine. The union of the passes is the candidate set, and each candidate keeps its rank
  in every pass as a feature. There is no Cartesian product.
- **Candidate pairs generated:** __NCAND__ for 1,732,544 test S1 entities (__CANDPER__ per S1).
- **How you ensured true matches were not lost:** every change was measured on a fixed training
  sample of 30,000 S1 entities searched against the **full** S2/S3. Recall of true pairs went
  from 0.69 (first version) to 0.84 (alignment fix), 0.955 (+ address pass) and 0.965
  (+ romanisation / skeleton / cleanup). The macro-F0.5 blocking ceiling reached 0.986–0.988.

---

## 4. Matching Model

**Features used (__NFEAT__):**
- Name features: word unigram / bigram Jaccard, phonetic-skeleton Jaccard, name TF-IDF cosine,
  Levenshtein ratio, rapidfuzz token-sort / token-set / partial ratios, core-name ratio (legal
  suffixes removed), first-token agreement, lengths.
- Address features: word unigram / bigram Jaccard, address TF-IDF cosine, token-set ratio,
  Levenshtein ratio, postal-code and house-number agreement, lengths.
- Candidate context: combined score, rank within source / overall, gap and ratio to the best
  candidate, candidate count, rank in each retrieval pass, source flag, country agreement.
- Cross-candidate consistency: max / mean full, name and address cosine between the candidate
  and the S1 entity's other top-5 candidates, count of those above 0.5, and name / address
  token-set ratio to the best other candidate.

**Model type:** LightGBM binary classifier (MIT licence), trained on __TRAINQ__ sampled training
S1 entities searched against the full S2/S3. Negatives are sampled 1:5, 60% of them hard (from
the entity's top 3). Early stopping uses held-out entities, then the model is refit on all
sampled rows.  
**Threshold selection method:** exact macro-F0.5 (singletons included) on 20% held-out
training entities with their full candidate lists and full ground truth (blocking misses count as
misses). A global probability threshold is combined with a one-to-one rule: each S2/S3 record
keeps only its most probable S1, as the ground truth is strictly one-to-one.

---

## 5. Results & Error Analysis

| Version | Blocking recall | Blocking ceiling | Validation F0.5 | Leaderboard |
|---|---|---|---|---|
| Baseline (word TF-IDF, before the alignment fix) | 0.695 | 0.815 | 0.7206 | 0.72 |
| + alignment fix, address pass, romanisation, skeleton, cleanup (D1) | 0.958 | 0.9855 | 0.9476 | – |
| + cross-candidate features, 250k training entities (D2) | __RECALL2__ | __CEIL2__ | __VAL_F05__ | – |

Local validation matched the leaderboard within 0.003 for the baseline.

- **F_0.5 Score (macro):** __VAL_F05__ (validation).
- **Common false positives (wrong merges):** different branches or businesses sharing a name
  where the target has an empty address (only the name can be compared); neighbouring house
  numbers on the same street (`1331` vs `1332 Andrews Street`); and invented names at exactly
  the same address (`Yumavio` at the S1 address).
- **Common false negatives (missed matches):** S1 records whose address is itself corrupted
  (`414 8th Street` vs the copies' `41 Eighth St`), domain-style names with weak addresses, and
  heavy typos in both fields at once. About 16% of the remaining loss is blocking misses; the
  rest is split between model misses and over-prediction.

---

## 6. Conclusion
Careful miss analysis showed that the loss came from blocking, not the classifier. One bug fix
and two cheap, data-driven blocking passes (address-only retrieval, and phonetic keys over
romanised text) raised the ceiling from 0.82 to 0.99. Validation F0.5 rose from 0.72 to
__VAL_F05__ with the same LightGBM model family. Lesson: measure the blocking ceiling first and
inspect raw misses before tuning the model.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/er_pipeline.py`: all pipeline logic (loading, romanisation, keys, multi-pass blocking,
  features, LightGBM, decision, writing and self-validation of outputs).
- `src/run_pipeline.py`: command line entry point.
- `src/decide.py`: optional per-entity expected-F0.5 selection from a run's cached probabilities.
- `src/make_dummy_data.py`: synthetic data for smoke tests.
- `README.md`, `requirements.txt`.

Reproduce `output/matching_results.tsv` and `output/candidate_pairs.tsv` (from `student_resource/`):
```bash
pip install -r code/business_entity_resolution/requirements.txt
__COMMAND__
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
Runtime is about __RUNTIME__ on 16 threads with 16 GB RAM (no GPU needed).

### B. Additional Results
Blocking experiments (30,000 training S1 entities, full S2/S3):

| Change | Recall of true pairs | Candidates per S1 |
|---|---|---|
| Word TF-IDF, K=19 (before fix) | 0.691 | 38 |
| + row-alignment fix | 0.8385 | 38 |
| + address-only pass, k=10 | 0.9554 | 52.6 |
| + address-only pass, k=20 | 0.9617 | 72.0 |
| + name-only pass, k=20 (no gain, dropped) | 0.9624 | 80.5 |
| + romanisation, skeleton, cleanup (address k=10 / 20) | 0.9588 / 0.9646 | 53.2 / 72.8 |

Fair play: no external data, APIs, geocoding or pretrained models; romanisation rules come only
from Unicode character names in the Python standard library. No ID or row-order signal is used,
and country is only compared for equality, so France (test only) is handled like any other
country.
