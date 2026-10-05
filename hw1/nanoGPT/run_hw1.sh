#!/bin/zsh
# Runs all HW1 experiments sequentially. Usage: ./run_hw1.sh [name ...]
cd "$(dirname "$0")"
PY=../.venv/bin/python
typeset -A RUNS
RUNS=(
  baseline ""
  rmsnorm  "--norm_type=rmsnorm"
  swiglu   "--mlp_type=swiglu"
  nope     "--pos_type=none"
  rope     "--pos_type=rope"
  gqa      "--n_kv_head=2"
)
if (( $# )); then names=($@); else names=(baseline rmsnorm swiglu nope rope gqa); fi
mkdir -p ../logs
for n in $names; do
  echo "=== $n ==="
  $PY -u train.py config/hw1_base.py --out_dir=out-hw1/$n ${=RUNS[$n]} > ../logs/$n.log 2>&1
  tail -1 ../logs/$n.log
done
