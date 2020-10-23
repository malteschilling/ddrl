#!/bin/bash
#SBATCH --job-name=ddrS_centr
#SBATCH --time=18:00:00
#SBATCH --mail-type=FAIL,END
#SBTACH --mail-user=mschilli@techfak.uni-bielefeld.de
#SBATCH --output=logs/job-exp_centralized-%j.log
#SBATCH --nodelist=schlaubi
#SBATCH --cpus-per-task=15

. $HOME/.bashrc
conda activate ray
export MUJOCO_PY_MJKEY_PATH="/media/compute/homes/mschilli/.mujoco/mjkey_schlaubi.txt"
cd $HOME/ddrl/ddrl/
module load cuda/10.1
srun python run_quantruped_multi_tune_baseline_hparam.py --policy_scope QuantrupedMultiEnv_Centralized
