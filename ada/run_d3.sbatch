#!/bin/bash
# Full run on the cluster. Submit from the repo root:   sbatch ada/run_d3.sbatch
# Override the run name / preset:                         RUN=D3 PRESET=d3 sbatch ada/run_d3.sbatch
#SBATCH -J er_d3
#SBATCH -c 40
#SBATCH --mem=60G
#SBATCH -t 07:00:00
#SBATCH -o logs/slurm_%x_%j.log
## Uncomment / edit if your cluster requires them:
##SBATCH -A <your_account>
##SBATCH -p <partition>
##SBATCH --gres=gpu:1          # some clusters only hand out CPUs together with a GPU (the GPU is not used)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs output artifacts

eval "$(conda shell.bash hook 2>/dev/null)" || source "$HOME/miniconda3/bin/activate"
conda activate er

RUN=${RUN:-D6}
PRESET=${PRESET:-d6}
THREADS=${SLURM_CPUS_PER_TASK:-40}
SRC=code/business_entity_resolution/src

BASE="--threads $THREADS --pool 150 --max-posting 10000 --k-max 20 --indic-translit --name-skeleton --name-clean"
V2="--norm-v2 --miss-features"
COMMON="$BASE --cross-features"
# d6 = final: decoy-aware address numbers, same-name sibling counts, target-side competition over ALL S1,
# reverse pass, test-like validation (threshold tuned with every S1 competing, like the leaderboard)
D6="$BASE $V2 --train-queries 400000 --k-addr-pass 20 --k-rev-pass 3 --house-fuzzy --addr-numbers --sib-features --target-features --rounds 3000 --max-train-rows 6000000"
case "$PRESET" in
  d6)   FLAGS="$D6" ;;
  d6x)  FLAGS="$D6 --cross-features" ;;
  d6s)  FLAGS="$BASE $V2 --train-queries 250000 --k-addr-pass 15 --k-rev-pass 3 --house-fuzzy --addr-numbers --sib-features --target-features --rounds 3000" ;;
  d2)   FLAGS="$COMMON --train-queries 250000 --k-addr-pass 15" ;;                                   # = D2 (val 0.9537)
  d3)   FLAGS="$COMMON --train-queries 400000 --k-addr-pass 20 --house-fuzzy --two-stage --rounds 3000 --max-train-rows 6000000" ;;
  d3s)  FLAGS="$COMMON --train-queries 250000 --k-addr-pass 15 --house-fuzzy --two-stage --rounds 3000" ;;  # smaller / faster d3
  d4)   FLAGS="$COMMON --train-queries 400000 --k-addr-pass 20 --house-fuzzy --two-stage --target-features --rounds 3000 --max-train-rows 6000000" ;;
  d4s)  FLAGS="$COMMON --train-queries 250000 --k-addr-pass 15 --house-fuzzy --two-stage --target-features --rounds 3000" ;;
  d5)   FLAGS="$COMMON --train-queries 400000 --k-addr-pass 20 --k-rev-pass 3 --house-fuzzy --two-stage --target-features --rounds 3000 --leaves 255 --max-train-rows 6000000" ;;
  d5s)  FLAGS="$COMMON --train-queries 250000 --k-addr-pass 15 --k-rev-pass 3 --house-fuzzy --two-stage --target-features --rounds 3000 --leaves 255" ;;
  *)    echo "unknown PRESET $PRESET"; exit 1 ;;
esac

echo "[$(date +%T)] RUN=$RUN PRESET=$PRESET threads=$THREADS"
echo "flags: $FLAGS"
export PYTHONIOENCODING=utf-8
python $SRC/run_pipeline.py --data-dir dataset --output-dir output/$RUN --model-dir artifacts/$RUN $FLAGS

python utils/validate_submission.py --matching output/$RUN/matching_results.tsv \
    --candidate output/$RUN/candidate_pairs.tsv --test-dir dataset/test

# decision rule tuned on the TEST-LIKE validation (threshold + ambiguous-sibling margin + per-source cap)
if [ -f artifacts/$RUN/cache/testlike_pairs.parquet ]; then
    python $SRC/decide_testlike.py --run $RUN --out output/${RUN}_tl --test-dir dataset/test &&     python utils/validate_submission.py --matching output/${RUN}_tl/matching_results.tsv         --candidate output/${RUN}_tl/candidate_pairs.tsv --test-dir dataset/test || true
    cat output/${RUN}_tl/decision_testlike.json 2>/dev/null || true
fi
grep -E '"testlike_f05|"testlike_threshold' artifacts/$RUN/run_report.json || true

# optional per-entity expected-F0.5 selection (writes output/${RUN}_ef only if it beats the threshold)
python $SRC/decide.py --run artifacts/$RUN --out output/${RUN}_ef --test-dir dataset/test || true
[ -f output/${RUN}_ef/matching_results.tsv ] && cp output/$RUN/candidate_pairs.tsv output/${RUN}_ef/ && \
    python utils/validate_submission.py --matching output/${RUN}_ef/matching_results.tsv \
        --candidate output/${RUN}_ef/candidate_pairs.tsv --test-dir dataset/test || true
grep -E '"val_f05"|val_f05_blocking_ceiling|val_f05_stage1_best' artifacts/$RUN/run_report.json || true
cat output/${RUN}_ef/decision.json 2>/dev/null || true
echo "[$(date +%T)] done"
