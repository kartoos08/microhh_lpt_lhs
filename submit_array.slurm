#!/bin/bash
#SBATCH --job-name=mh_lpt_lhs
#SBATCH --array=0-99%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CASE=$(printf "case_%03d" "${SLURM_ARRAY_TASK_ID}")
CDIR="${ROOT}/cases/${CASE}"
mkdir -p "${ROOT}/slurm"
cd "${CDIR}"

MICROHH_BIN="${MICROHH_BIN:-/path/to/build/microhh}"

# Current upstream MicroHH supports CUDA and MPI separately, not together;
# this array assumes one GPU / one MicroHH process per case.
"${MICROHH_BIN}" init "${CASE}"
"${MICROHH_BIN}" run  "${CASE}"
