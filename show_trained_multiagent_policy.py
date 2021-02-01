import ray
import pickle5 as pickle
import os

from ray.tune.registry import get_trainable_cls
from ray.rllib.evaluation.worker_set import WorkerSet

import simulation_envs
import models
from evaluation.rollout_episodes import rollout_episodes

"""
    Visualizing a learned (multiagent) controller,
    for evaluation or visualisation.
    
    This is adapted from rllib's rollout.py
    (github.com/ray/rllib/rollout.py)
"""

# Setting number of steps and episodes
num_steps = int(400)
num_episodes = int(1)

# Selecting checkpoint to load
config_checkpoint = "/Users/mschilling/Desktop/gpu_cluster/ray_results_12_09/HF_10_QuantrupedMultiEnv_Centralized/PPO_QuantrupedMultiEnv_Centralized_989cd_00000_0_2020-12-08_18-34-16/checkpoint_1250/checkpoint-1250"
#config_checkpoint = "/Users/mschilling/Desktop/gpu_cluster/ray_results_12_09/HF_10_QuantrupedMultiEnv_Local/PPO_QuantrupedMultiEnv_Local_1a49c_00000_0_2020-12-04_12-08-56/checkpoint_1250/checkpoint-1250"
#config_checkpoint = "/Users/mschilling/ray_results/Hexa_HexapodMultiEnv_Centralized/PPO_HexapodMultiEnv_Centralized_09ee7_00000_0_2021-01-06_15-54-12/checkpoint_625/checkpoint-625"
#"/Users/mschilling/Desktop/gpu_cluster/ray_results/exp_QuantrupedMultiEnv_Centralized/PPO_QuantrupedMultiEnv_Centralized_6e846_00000_0_2020-10-24_15-47-18/checkpoint_2084/checkpoint-2084"
#config_checkpoint="/Users/mschilling/Desktop/develop/Decentralized_DRL/ray_results/rllib_centralized_2/PPO_QuantrupedMultiEnv_Centralized_7443a_00000_0_2020-10-21_20-23-02/checkpoint_3125/checkpoint-3125"
#config_checkpoint="/Users/mschilling/ray_results/rllib_quantruped/PPO_QuAntruped-v3_e38c0_00000_0_2020-10-14_11-51-53/checkpoint_625/checkpoint-625"
#config_checkpoint = "/Users/mschilling/ray_results/exp_distRew_QuantrupedMultiEnv_Local/PPO_QuantrupedMultiEnv_Local_d6587_00000_0_2020-10-26_10-44-27/checkpoint_4168/checkpoint-4168"
config_dir = os.path.dirname(config_checkpoint)
config_path = os.path.join(config_dir, "params.pkl")

# Loading configuration for checkpoint.
if not os.path.exists(config_path):
    config_path = os.path.join(config_dir, "../params.pkl")
if os.path.exists(config_path):
    with open(config_path, "rb") as f:
        config = pickle.load(f)
        
# Starting ray and setting up ray.
if "num_workers" in config:
    config["num_workers"] = min(2, config["num_workers"])
ray.init()
cls = get_trainable_cls('PPO')
# Setting config values (required for compatibility between versions)
config["create_env_on_driver"] = True
config['env_config']['hf_smoothness'] = 1.
if "no_eager_on_workers" in config:
    del config["no_eager_on_workers"]

# Load state from checkpoint.
agent = cls(env=config['env'], config=config)
agent.restore(config_checkpoint)

# Retrieve environment for the trained agent.
if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
    env = agent.workers.local_worker().env
    
# Rolling out simulation = stepping through simulation. 
rollout_episodes(env, agent, num_episodes=num_episodes, num_steps=num_steps, render=True)
agent.stop()