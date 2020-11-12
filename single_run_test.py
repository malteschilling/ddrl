import ray
import pickle5 as pickle
import os
import numpy as np
import argparse

from ray.tune.registry import get_trainable_cls
from ray.rllib.evaluation.worker_set import WorkerSet

import simulation_envs
import models
from rollout_episodes import rollout_episodes

parser = argparse.ArgumentParser()
parser.add_argument("--ray_results_dir", required=False)
args = parser.parse_args()
if 'ray_results_dir' in args and args.ray_results_dir: 
    ray_results_dir = args.ray_results_dir
else:
     ray_results_dir = os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_Centralized'

# exp_path = [os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_Centralized',
#     os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_FullyDecentral',
#     os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_Local',
#     os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_SingleDiagonal',
#     os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_SingleNeighbor',
#     os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_TwoDiags',
#     os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_TwoSides',
#     os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_QuantrupedMultiEnv_SingleToFront']
        
experiment_dirs = [os.path.join(ray_results_dir,dI) for dI in os.listdir(ray_results_dir) if os.path.isdir(os.path.join(ray_results_dir,dI))]
exp_params = [os.path.join(exp_d, 'params.pkl') for exp_d in experiment_dirs]
exp_checkpoint = [os.path.join(exp_d, 'checkpoint_624', 'checkpoint-624') for exp_d in experiment_dirs]

ray.init()

all_rew = []
all_cot = []
all_vel = []

with open(exp_params[0], "rb") as f:
    config = pickle.load(f)
if "num_workers" in config:
    config["num_workers"] = min(2, config["num_workers"])
cls = get_trainable_cls('PPO')
agent = cls(env=config['env'], config=config)
# Load state from checkpoint.
agent.restore(exp_checkpoint[0])

if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
    env = agent.workers.local_worker().env
res_rollout = rollout_episodes(env, agent, num_episodes=4, num_steps=1000, render=True)
eval_rew = np.array(res_rollout[0])
all_rew.append(eval_rew)
eval_cot = np.array(res_rollout[1])
all_cot.append(eval_cot)
eval_vel = np.array(res_rollout[2])
all_vel.append(eval_vel)