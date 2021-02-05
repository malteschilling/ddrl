import ray
import pickle5 as pickle
import os
import numpy as np
import argparse

import pandas as pd

from ray.tune.registry import get_trainable_cls
from ray.rllib.evaluation.worker_set import WorkerSet

import target_envs
import models
from evaluation.rollout_episodes_compute_gradient import rollout_episodes

"""
    Generating Importance Matrix - section 4.8
    
    This shows how a small change in one input feature dimension affects the 
    control signals for the centralized control architecture at end of training.
    
    As we are dealing with DRL, we are accumulating this gradient over multiple rollouts 
    and runs of the agent in the environment. 
    Each seed from experiment three of the centralized architecture was run for ten 
    evaluation runs (producing multiple trajectories) of each t=1000 control steps. 
    
    We are interested in how a change in the input (observation over all i=44 dimensions) 
    changes the output of the network (mean value of the normal distribution 
    from which actions are sampled).
    
    For measuring this gradient, we recorded the original output of the policy network 
    during each control step and computed the gradient numerically with respect to the 
    individual input dimensions of the observation space. 
    As the different input dimensions are of quite different scale and variation during 
    walking, we had to normalize the computed gradient for which we used the standard 
    deviation of the particular feature dimension (measured over time as sigma_i).
"""

parser = argparse.ArgumentParser()
parser.add_argument("--hf_smoothness", required=False)
args = parser.parse_args()

if args.hf_smoothness is not None: 
    hf_smoothness_eval = float(args.hf_smoothness)
else:
    hf_smoothness_eval = 0.8

# Path to trained models.
exp_path = [os.getcwd() + '/Results/experiment_3_models_curriculum_tvel//Tvel_QuantrupedMultiEnv_Centralized_TVel']
experiment_dirs = [[os.path.join(exp_path_item,dI) for dI in os.listdir(exp_path_item) if os.path.isdir(os.path.join(exp_path_item,dI))] for exp_path_item in exp_path]

ray.init()

df = pd.DataFrame([], columns=["approach", "seed", "trained_on", "evaluated_on", "target_velocity", "simulation_run", "reward", "duration", "distance", "power", "velocity", "CoT" ])

exp_it = 0
for exp_dir in experiment_dirs:
    exp_params = [os.path.join(exp_d, 'params.pkl') for exp_d in exp_dir]
    exp_checkpoint = [os.path.join(exp_d, 'checkpoint_1250', 'checkpoint-1250') for exp_d in exp_dir]

    all_rew = []
    all_steps = []
    all_dist = []
    all_power_total = []
    all_cot = []
    all_vel = []

    for experiment in range(0, len(exp_params) ):
        #with open(exp_params[experiment], "rb") as f:
         #   config = pickle.load(f)
        try:
            with open(exp_params[experiment], "rb") as f:
                config = pickle.load(f)
        except:
            with open(exp_dir[experiment] + "/../params_py36.pkl", "rb") as f:
                config = pickle.load(f)
        if "num_workers" in config:
            config["num_workers"] = min(2, config["num_workers"]) 
        config['env_config']['target_velocity'] = 1.0
        config["create_env_on_driver"] = True
        config['env_config']['hf_smoothness'] = hf_smoothness_eval
        if "no_eager_on_workers" in config:
            del config["no_eager_on_workers"]
        cls = get_trainable_cls('PPO')
        agent = cls(env=config['env'], config=config)
        # Load state from checkpoint.
        agent.restore(exp_checkpoint[experiment])
        if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
            env = agent.workers.local_worker().env
    
        # Call rollouts that collect gradients along the trajectory as well.
        res_rollout = rollout_episodes(env, agent, num_episodes=10, num_steps=1000, render=False, experiment_nr=experiment)

        # Write detailed data to panda file
        for sim_it in range(0, len(res_rollout[0])):
            new_pd_entry = pd.Series({"approach": exp_path[exp_it].split('_')[-2], 
                "seed": exp_params[experiment].split('0000')[-1][0], 
                "trained_on": "flat", 
                "evaluated_on": hf_smoothness_eval,
                "target_velocity": config['env_config']['target_velocity'],
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
        print('Mean for ', exp_path[exp_it].split('_')[-2], '/', exp_params[run_it].split('0000')[-1][0], f': {np.mean(eval_results):.2f}, std.dev.: {np.std(eval_results):.2f}, Distance: {np.mean(all_dist[run_it]):.2f}, Steps: {np.mean(all_steps[run_it]):.2f}')
    
        run_it += 1
    #    print('Overall Mean for ', exp_path[exp_it].split('_')[-1], f': {np.mean(all_rew):.2f}, std.dev.: {np.std(all_rew):.2f}; CoT: {np.mean(all_cot):.2f}; Vel.: {np.mean(all_vel):.2f}, {np.std(all_vel):.2f}')
    #   print(exp_path[exp_it].split('_')[-1], f' && {np.mean(all_rew):.2f} & ({np.std(all_rew):.2f}) && {np.mean(all_vel):.2f} & ({np.std(all_vel):.2f}) & {np.mean(all_cot):.2f}')
    print('Overall Mean for ', exp_path[exp_it].split('_')[-2], f': {np.mean(all_rew):.2f}, std.dev.: {np.std(all_rew):.2f}; CoT: {np.mean(all_cot):.2f}; Vel.: {np.mean(np.sum(all_dist)/np.sum(all_steps)):.2f}')
    print(exp_path[exp_it].split('_')[-1], f' && {np.mean(all_rew):.2f} & ({np.std(all_rew):.2f}) && {np.mean(np.sum(all_dist)/np.sum(all_steps))} & ({np.std(all_vel):.2f}) & {np.mean(all_cot):.2f}')
    exp_it += 1
    df.to_csv("evaluation_tvel1_smooth08_testgrad_" + str(hf_smoothness_eval) + ".csv")