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
tail -f logs/slurm_er_d3_<jobid>.log         # live log; Ctrl+C stops watching, not the job
```
If `sbatch` rejects the job, open `ada/run_d3.sbatch` and uncomment the `-A` (account),
`-p` (partition) or `--gres=gpu:1` lines as your cluster requires.

Presets (default `RUN=D6 PRESET=d6`):
- `sbatch ada/run_d3.sbatch`: **final (d6)**. Decoy-aware address-number features, same-name sibling counts,
  target-side competition over all S1, reverse pass, and a **test-like validation** (every S1 competes, like the
  leaderboard) that sets the threshold. Afterwards `decide_testlike.py` tunes a margin/cap rule and writes
  `output/D6_tl`. About 2–2.5 h, ~30 GB.
- `RUN=D6x PRESET=d6x sbatch ada/run_d3.sbatch`: the same plus cross-candidate features.
- `RUN=D6s PRESET=d6s sbatch ada/run_d3.sbatch`: 250k queries, address pass 15 (faster).
- Older presets: d2, d3, d3s, d4, d4s, d5, d5s.

Useful log lines: `testlike_f05` (the number to trust: it mimics the leaderboard), `[test] candidates`,
`test features`, `VALIDATION PASS`, `best test-like F0.5 ... (thr, margin, cap)`, `done`.

## 5. On your laptop: copy the results back (~1.5 GB)
```powershell
scp -r <user>@ada.iiit.ac.in:~/ml-am/output/D6_tl output/
scp <user>@ada.iiit.ac.in:~/ml-am/artifacts/D6/run_report.json artifacts/D6_run_report.json
```
Submit `output/D6_tl` (tuned decision rule) or `output/D6`. Its `testlike_f05` is the leaderboard estimate;
D1 scored 0.936 on the leaderboard and remains the fallback.
