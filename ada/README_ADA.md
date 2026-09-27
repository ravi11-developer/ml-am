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

Presets: `PRESET=d3` (400k queries, full), `PRESET=d3s` (250k queries, faster), `PRESET=d2` (reproduces
D2). Example: `RUN=D3s PRESET=d3s sbatch ada/run_d3.sbatch`.

Useful log lines: `stage-1 best val F0.5` vs `val_f05` (the stage-2 gain), `[test] candidates`,
`test s2 features`, `VALIDATION PASS`, `done`.

## 5. On your laptop: copy the results back (~1.5 GB)
```powershell
scp -r <user>@ada.iiit.ac.in:~/ml-am/output/D3 output/
scp <user>@ada.iiit.ac.in:~/ml-am/artifacts/D3/run_report.json artifacts/D3_run_report.json
```
Submit `output/D3` (or `output/D3_ef` if `decision.json` shows a higher `ef_val_f05`) only if its
`val_f05` beats D2's 0.9537. Otherwise keep D2.
