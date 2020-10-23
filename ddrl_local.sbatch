#!/bin/bash
#SBATCH --job-name=ddrl_fd
#SBATCH --time=22:00:00
#SBATCH --mail-type=FAIL,END
#SBTACH --mail-user=mschilli@techfak.uni-bielefeld.de
#SBATCH --output=logs/job-centr-%j.log
#SBATCH --nodelist=handy
#SBATCH --cpus-per-task=15

. $HOME/.bashrc
conda activate ray
cd $HOME/ddrl/ddrl/
module load cuda/10.1
srun python run_quantruped_multi_tune_baseline_hparam.py --policy_scope QuantrupedMultiEnv_Local
