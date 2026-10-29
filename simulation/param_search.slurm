#!/bin/bash
#SBATCH --cpus-per-task 2
#SBATCH -e /home/murban/reasondb/logging/slurm/slurm-%A-%a.err
#SBATCH -o /home/murban/reasondb/logging/slurm/slurm-%A-%a.out

python simulation/guarantee_pushdown_test.py --storage "postgresql://murban@faststore01.tuda.systems/dgx_murban" --worker-id ${SLURM_ARRAY_TASK_ID} --search --n-steps 300
