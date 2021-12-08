import ray
import pickle
# When using older versions of python (3.6 <=), use pickle5 when you want to interchange
# saved picklefiles
#import pickle5 as pickle
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
num_steps = int(200)
num_episodes = int(1)

ray.init()

smoothness = 1.0

# Selecting checkpoint to load
config_checkpoints = [os.getcwd() + "/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_Centralized/PPO_QuantrupedMultiEnv_Centralized_989cd_00006_6_2020-12-08_18-34-17/checkpoint_1250/checkpoint-1250",
    os.getcwd() + "/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_Local/PPO_QuantrupedMultiEnv_Local_1a49c_00005_5_2020-12-05_06-20-13/checkpoint_1250/checkpoint-1250",
    os.getcwd() + "/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_TwoSides/PPO_QuantrupedMultiEnv_TwoSides_6654b_00003_3_2020-12-06_02-21-44/checkpoint_1250/checkpoint-1250",
    os.getcwd() + "/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_FullyDecentral/PPO_QuantrupedMultiEnv_FullyDecentral_19697_00003_3_2020-12-04_12-08-56/checkpoint_1250/checkpoint-1250"]
    
# Selecting checkpoint to load for smoothness 0.8
config_checkpoints = [os.getcwd() + "/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_Centralized/PPO_QuantrupedMultiEnv_Centralized_989cd_00006_6_2020-12-08_18-34-17/checkpoint_1250/checkpoint-1250",
    os.getcwd() + "/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_Local/PPO_QuantrupedMultiEnv_Local_1a49c_00003_3_2020-12-04_12-08-57/checkpoint_1250/checkpoint-1250",
    os.getcwd() + "/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_TwoSides/PPO_QuantrupedMultiEnv_TwoSides_6654b_00006_6_2020-12-06_17-42-00/checkpoint_1250/checkpoint-1250",
    os.getcwd() + "/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_FullyDecentral/PPO_QuantrupedMultiEnv_FullyDecentral_19697_00004_4_2020-12-04_12-08-56/checkpoint_1250/checkpoint-1250"]

# Selecting checkpoint to load for smoothness 0.6
#config_checkpoints = [os.getcwd() + "/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_Centralized/PPO_QuantrupedMultiEnv_Centralized_989cd_00007_7_2020-12-08_18-34-17/checkpoint_1250/checkpoint-1250",
config_checkpoints = [os.getcwd() + "/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_Local/PPO_QuantrupedMultiEnv_Local_1a49c_00005_5_2020-12-05_06-20-13/checkpoint_1250/checkpoint-1250",
#    os.getcwd() + "/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_TwoSides/PPO_QuantrupedMultiEnv_TwoSides_6654b_00006_6_2020-12-06_17-42-00/checkpoint_1250/checkpoint-1250",
    os.getcwd() + "/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_FullyDecentral/PPO_QuantrupedMultiEnv_FullyDecentral_19697_00003_3_2020-12-04_12-08-56/checkpoint_1250/checkpoint-1250"]
    
# Selecting checkpoint to load
#config_checkpoints = [os.getcwd() + "/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_Centralized/PPO_QuantrupedMultiEnv_Centralized_989cd_00006_6_2020-12-08_18-34-17/checkpoint_1250/checkpoint-1250",
 #   os.getcwd() + "/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_Local/PPO_QuantrupedMultiEnv_Local_1a49c_00005_5_2020-12-05_06-20-13/checkpoint_1250/checkpoint-1250",
config_checkpoints = [os.getcwd() + "/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_TwoSides/PPO_QuantrupedMultiEnv_TwoSides_6654b_00003_3_2020-12-06_02-21-44/checkpoint_1250/checkpoint-1250",
    os.getcwd() + "/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_FullyDecentral/PPO_QuantrupedMultiEnv_FullyDecentral_19697_00003_3_2020-12-04_12-08-56/checkpoint_1250/checkpoint-1250"]
    
# Afterwards put together using
# ffmpeg -framerate 20 -pattern_type glob -i '*.png' -filter:v scale=720:-1 -vcodec libx264 -pix_fmt yuv420p -g 1 out.mp4

#HF_10_QuantrupedMultiEnv_Local/PPO_QuantrupedMultiEnv_Local_1a49c_00003_3_2020-12-04_12-08-57
#HF_10_QuantrupedMultiEnv_TwoSides/PPO_QuantrupedMultiEnv_TwoSides_6654b_00006_6_2020-12-06_17-42-00
#HF_10_QuantrupedMultiEnv_FullyDecentral/PPO_QuantrupedMultiEnv_FullyDecentral_19697_00004_4_2020-12-04_12-08-56

for config_checkpoint in config_checkpoints:
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
    cls = get_trainable_cls('PPO')
    # Setting config values (required for compatibility between versions)
    config["create_env_on_driver"] = True
    config['env_config']['hf_smoothness'] = smoothness
    if "no_eager_on_workers" in config:
        del config["no_eager_on_workers"]

    # Load state from checkpoint.
    agent = cls(env=config['env'], config=config)
    agent.restore(config_checkpoint)

    # Retrieve environment for the trained agent.
    if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
        env = agent.workers.local_worker().env

    save_image_dir = 'videos/' + config_path.partition('MultiEnv_')[2].partition('/')[0] + '_smoothn_' + str(smoothness)
    os.mkdir(save_image_dir)
    # Rolling out simulation = stepping through simulation. 
    rollout_episodes(env, agent, num_episodes=num_episodes, num_steps=num_steps, render=True, save_images=save_image_dir+"/img_", save_obs=save_image_dir)
    agent.stop()