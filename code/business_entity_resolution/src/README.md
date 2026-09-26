# Business Entity Resolution — reproduction instructions

## Layout
```
student_resource/
├── dataset/train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
├── dataset/test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
├── utils/validate_submission.py
└── code/business_entity_resolution/
    ├── src/run_pipeline.py      # full pipeline (same code as the notebook)
    ├── README.md
    └── requirements.txt
```

## Environment
```bash
conda create -n amazon-er python=3.11 -y
conda activate amazon-er
pip install -r code/business_entity_resolution/requirements.txt
```

## Run end to end (data -> blocking -> matching -> output)
From the `student_resource/` folder:
```bash
python code/business_entity_resolution/src/run_pipeline.py
```
The script finds `student_resource/` automatically when run from inside it. From anywhere else, set `ER_BASE`:
- Linux/macOS: `ER_BASE=/path/to/student_resource python run_pipeline.py`
- Windows (PowerShell): `$env:ER_BASE="C:\path\to\student_resource"; python run_pipeline.py`

Outputs: `output/matching_results.tsv` and `output/candidate_pairs.tsv`, then the official validator runs automatically.
Models and settings are saved to `artifacts/`. Seed is fixed (42), so reruns give the same results.

## Method in one paragraph
Names and addresses are normalized with hand-written rules (legal suffixes, abbreviations, accents, postal codes).
Candidates come from five blocking methods run within each country label: TF-IDF character n-gram top-K on names,
addresses, and both combined, a reverse name search, and postal-code + rare-token keys; very common n-grams are ignored
and at most 30 candidates are kept per Source 1 record. About 60 pair features (TF-IDF cosines, rapidfuzz string
similarities, postal/house-number agreement and conflicts, legal-suffix agreement, and rank/margin of each candidate
within its group) feed a LightGBM classifier trained with 3-fold GroupKFold by Source 1 entity. Out-of-fold
probabilities are isotonic-calibrated, each Source 2/3 record is kept only on its most likely Source 1 entity, and a
probability threshold plus maximum matches per entity are chosen to maximize macro F0.5. No external data is used.
