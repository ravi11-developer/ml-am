# Running on the Ada cluster

The GPU is not used. What matters is 40 CPU cores and about 60 GB RAM. Runs D3 (two-stage model)
in about 1.5–2.5 h.

## 1. On your laptop: push the code
```powershell
git push origin master
```

## 2. On Ada (login node): clone and set up (~10 min, once)
```bash
ssh <user>@ada.iiit.ac.in
git clone https://github.com/ravi11-developer/ml-am.git
cd ml-am
bash ada/setup_env.sh                   # conda env "er" with Python 3.11 + pinned packages
```

## 3. On your laptop: copy the data and validator (not in git), ~2.4 GB
Run from the `student_resource` folder on the laptop. Use the same path as the clone above:
```powershell
scp -r dataset utils <user>@ada.iiit.ac.in:~/ml-am/
```

## 4. On Ada: submit and watch
```bash
sbatch ada/run_d3.sbatch                     # PRESET=d3 (default), RUN=D3
squeue -u $USER                              # job state (PD = waiting, R = running)
tail -f logs/slurm_er_<jobid>.log         # live log; Ctrl+C stops watching, not the job
```
If `sbatch` rejects the job, open `ada/run_d3.sbatch` and uncomment the `-A` (account),
`-p` (partition) or `--gres=gpu:1` lines as your cluster requires.

Presets (default `RUN=D7 PRESET=d7`):
- `sbatch ada/run_d3.sbatch`: **recommended (d7)**. Decoy-aware address numbers, same-name sibling counts,
  missing-rare-token features, French-aware normalisation (norm-v2), house-number fuzzy match, address pass 20,
  reverse pass, 400k training queries. The feature set was validated on real data (30k check 0.9424 -> 0.9598).
  About 1.5–2 h, ~25 GB.
- `RUN=D7s PRESET=d7s sbatch ada/run_d3.sbatch`: 250k queries, address pass 15 (faster, ~1.2 h).
- `RUN=D6 PRESET=d6 sbatch ada/run_d3.sbatch`: d7 plus target-side competition over all S1 and an in-pipeline
  test-like validation (+30–40 min, untested at full scale).

Useful log lines: `testlike_f05` (the number to trust: it mimics the leaderboard), `[test] candidates`,
`test features`, `VALIDATION PASS`, `best test-like F0.5 ... (thr, margin, cap)`, `done`.

## 5. On your laptop: copy the results back (~1.5 GB)
```powershell
scp -r <user>@ada.iiit.ac.in:~/ml-am/output/D7 output/
scp <user>@ada.iiit.ac.in:~/ml-am/artifacts/D7/run_report.json artifacts/D7_run_report.json
```
Submit `output/D7` (tuned decision rule) or `output/D6`. Its `testlike_f05` is the leaderboard estimate;
D1 scored 0.936 on the leaderboard and remains the fallback.
