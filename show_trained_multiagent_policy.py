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
num_steps = int(500)
num_episodes = int(1)

# Selecting checkpoint to load
config_checkpoint = "Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_Centralized/PPO_QuantrupedMultiEnv_Centralized_989cd_00000_0_2020-12-08_18-34-16/checkpoint_1250/checkpoint-1250"
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