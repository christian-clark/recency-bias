#!/bin/bash
#SBATCH --partition=yourPartition
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --mem=300gb

set -x
set -e

# Some potentially useful distributed environment variables
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# Your hostfile creation script from above
./write_hostfile.sh
# Tell DeepSpeed where to find our generated hostfile via DLTS_HOSTFILE
export DLTS_HOSTFILE=/sample/path/to/hostfiles/hosts_$SLURM_JOBID

DEEPY=/path/to/gpt-neox/deepy.py
TRAINSLM=/path/to/gpt-neox/train_slms.py
CONFIG=/path/to/gpt-neox/configs/pythia-2-4-256-1k.yml

python $DEEPY $TRAINSLM $CONFIG

