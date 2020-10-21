import numpy as np
import gym
from gym import spaces

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from ray import tune
from ray.tune import grid_search
import time

import simulation_envs
import models
from simulation_envs.quantruped_adaptor_multi_environment import QuantrupedMultiEnv_Centralized

ray.init(ignore_reinit_error=True)

config = ppo.DEFAULT_CONFIG.copy()

config['env'] = "QuantrupedMultiEnv_Centralized"

# Baseline Defaults:
# nsteps=2048, mujoco: nsteps=2048,
#config['horizon']=2048 ???
config['num_envs_per_worker']=4
# 4 environments mult sgd_minibatch_size
#config['train_batch_size'] = 8192

# mujoco: nminibatches=32,
# nsteps  * nenvs/ nminibatches=32 
#config['sample_batch_size'] = 1600
config['sgd_minibatch_size'] = 2048 # Default: 128, or horizon?
config['num_sgd_iter'] = 10

# mujoco: lam=0.95,
config['lambda'] = 0.95        
# ent_coef=0.0, 
config['entropy_coeff'] = 0.
# lr=3e-4,
config['lr'] = 3e-4
# vf_coef=0.5,
config['vf_loss_coeff'] = 0.5
# max_grad_norm=0.5, 
config['grad_clip']=0.5
# gamma=0.99, 
config['gamma'] = 0.99
# nminibatches=4?
# noptepochs=4, mujoco: noptepochs=10,
#cliprange=0.2,
config['clip_param'] = 0.2


#config['seed'] = seed
config['vf_clip_param'] = 4000.

config['train_batch_size'] = 4000 #grid_search([4000, 65536]  

config['observation_filter'] = 'MeanStdFilter' #grid_search(['MeanStdFilter', 'NoFilter'])

config['model']['custom_model'] = "fc_glorot_uniform_init"
config['model']['fcnet_hiddens'] = [64, 64]
#config['model']['vf_share_layers'] = grid_search(['False', ' True'])

config['seed'] = round(time.time())

single_env = gym.make("QuAntruped-v3")
policies = QuantrupedMultiEnv_Centralized.return_policies(single_env.observation_space)

config["multiagent"] = {
        "policies": policies,
        "policy_mapping_fn": QuantrupedMultiEnv_Centralized.policy_mapping_fn,
        "policies_to_train": [list(policies.keys())[0]], #, "dec_B_policy"],
    }

analysis = tune.run(
      "PPO",
      name="rllib_git",
      num_samples=1,
      checkpoint_at_end=True,
      stop={"timesteps_total": 1000000},
      config=config,
  )
