import ray
import pickle5 as pickle
import os
import numpy as np
import argparse

import pandas as pd

from ray.tune.registry import get_trainable_cls
from ray.rllib.evaluation.worker_set import WorkerSet

import simulation_envs
import models
from evaluation.rollout_episodes import rollout_episodes

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

exp_path = [os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_Centralized',
    os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_FullyDecentral',
    os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_Local',
    os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_SingleDiagonal',
    os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_SingleNeighbor',
    os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_TwoDiags',
    os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_TwoSides',
    os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_QuantrupedMultiEnv_SingleToFront']
    
exp_path = [os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_Centralized',
    os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_FullyDecentral']
        
experiment_dirs = [[os.path.join(exp_path_item,dI) for dI in os.listdir(exp_path_item) if os.path.isdir(os.path.join(exp_path_item,dI))] for exp_path_item in exp_path]

ray.init()

# Set panda DataFrame structure
df = pd.DataFrame([], columns=["approach", "seed", "trained_on", "simulation_run", "reward", "duration", "distance", "power", "velocity", "CoT" ])

exp_it = 0
for exp_dir in experiment_dirs:
    print("EVALUATE EXPERIMENT: ", exp_path[exp_it].split('_')[-1])
    exp_params = [os.path.join(exp_d, 'params.pkl') for exp_d in exp_dir]
    exp_checkpoint = [os.path.join(exp_d, 'checkpoint_624', 'checkpoint-624') for exp_d in exp_dir]

    all_rew = []
    all_steps = []
    all_dist = []
    all_power_total = []
    all_cot = []
    all_vel = []
    
    for experiment in range(0, len(exp_params) ):
        with open(exp_params[experiment], "rb") as f:
            config = pickle.load(f)
        if "num_workers" in config:
            config["num_workers"] = min(2, config["num_workers"])
        cls = get_trainable_cls('PPO')
        agent = cls(env=config['env'], config=config)
        # Load state from checkpoint.
        agent.restore(exp_checkpoint[experiment])
        if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
            env = agent.workers.local_worker().env
        res_rollout = rollout_episodes(env, agent, num_episodes=10, num_steps=100, render=False)
        
        # Write detailed data to panda file
        for sim_it in range(0, len(res_rollout[0])):
            new_pd_entry = pd.Series({"approach": exp_path[exp_it].split('_')[-1], 
                "seed": exp_params[experiment].split('0000')[-1][0], 
                "trained_on": "flat", 
                "simulation_run": sim_it, 
                "reward": res_rollout[0][sim_it], 
                "duration": res_rollout[1][sim_it], 
                "distance": res_rollout[2][sim_it], 
                "power": res_rollout[3][sim_it],
                "velocity": res_rollout[4][sim_it],  
                "CoT": res_rollout[5][sim_it] })
            df = df.append(new_pd_entry, ignore_index=True)
        
        eval_rew = np.array(res_rollout[0])
        all_rew.append(eval_rew)
        eval_steps = np.array(res_rollout[1])
        all_steps.append(eval_steps)
        eval_dist = np.array(res_rollout[2])
        all_dist.append(eval_dist)
        eval_power_total = np.array(res_rollout[3])
        all_power_total.append(eval_power_total)
        eval_vel = np.array(res_rollout[4])
        all_vel.append(eval_vel)
        eval_cot = np.array(res_rollout[5])
        all_cot.append(eval_cot)
        
        #print('Mean for ', ray_results_dir.split('_')[-1], '/', exp_params[experiment].split('0000')[-1][0], f': {np.mean(eval_rew):.2f}, std.dev.: {np.std(eval_rew):.2f}')
        agent.stop()
            
    run_it = 0
    for eval_results in all_rew:
        print('Mean for ', exp_path[exp_it].split('_')[-1], '/', exp_params[run_it].split('0000')[-1][0], f': {np.mean(eval_results):.2f}, std.dev.: {np.std(eval_results):.2f}, Distance: {np.mean(all_dist[run_it]):.2f}, Steps: {np.mean(all_steps[run_it]):.2f}')
        
        run_it += 1
#    print('Overall Mean for ', exp_path[exp_it].split('_')[-1], f': {np.mean(all_rew):.2f}, std.dev.: {np.std(all_rew):.2f}; CoT: {np.mean(all_cot):.2f}; Vel.: {np.mean(all_vel):.2f}, {np.std(all_vel):.2f}')
 #   print(exp_path[exp_it].split('_')[-1], f' && {np.mean(all_rew):.2f} & ({np.std(all_rew):.2f}) && {np.mean(all_vel):.2f} & ({np.std(all_vel):.2f}) & {np.mean(all_cot):.2f}')
    print('Overall Mean for ', exp_path[exp_it].split('_')[-1], f': {np.mean(all_rew):.2f}, std.dev.: {np.std(all_rew):.2f}; CoT: {np.mean(all_cot):.2f}; Vel.: {np.mean(np.sum(all_dist)/np.sum(all_steps)):.2f}')
    print(exp_path[exp_it].split('_')[-1], f' && {np.mean(all_rew):.2f} & ({np.std(all_rew):.2f}) && {np.mean(np.sum(all_dist)/np.sum(all_steps))} & ({np.std(all_vel):.2f}) & {np.mean(all_cot):.2f}')
    exp_it += 1
df.to_csv("evaluation_" + "flat" + ".csv")