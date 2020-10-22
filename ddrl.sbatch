#!/bin/bash
#SBATCH --job-name=ddrl_centr
#SBATCH --time=2-12:00:00
#SBATCH --mail-type=FAIL,END
#SBTACH --mail-user=mschilli@techfak.uni-bielefeld.de
#SBATCH --output=logs/job-%j.log
#SBATCH --nodelist=handy
#SBATCH --cpus-per-task=17
#SBATCH --gres=gpu:gtx:1
#SBATCH --partition=cpu
. $HOME/.bashrc

source /media/compute/homes/mschilli/miniconda3/bin/activate ray
cd $HOME/ddrl/ddrl/
module load cuda/10.1
srun python run_quantruped_multi_tune_baseline_hparam.py