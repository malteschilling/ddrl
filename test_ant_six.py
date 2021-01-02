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

import hexapod_envs
import models

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--policy_scope", required=False)
args = parser.parse_args()
# Possible values: "QuantrupedMultiEnv_FullyDecentral", "QuantrupedMultiEnv_SingleNeighbor",
# "QuantrupedMultiEnv_SingleDiagonal", "QuantrupedMultiEnv_Local"
# if 'policy_scope' in args and args.policy_scope: 
#     policy_scope = args.policy_scope
# else:
#     policy_scope = 'HexapodMultiEnv_Centralized'
#  
# # To run: SingleDiagonal, SingleToFront, TwoSides, TwoDiags
# if policy_scope=="HexapodMultiEnv_FullyDecentral":
#     from hexapod_envs.hexapod_sixDecentralizedController_environments import HexapodFullyDecentralizedEnv as HexapodEnv
# elif policy_scope=="HexapodMultiEnv_Local":
#     from hexapod_envs.hexapod_sixDecentralizedController_environments import Hexapod_Local_Env as HexapodEnv
# elif policy_scope=="HexapodMultiEnv_TwoSides":
#     from hexapod_envs.hexapod_twoDecentralizedController_environments import Hexapod_TwoSideControllers_Env as HexapodEnv
# else:
#     from hexapod_envs.hexapod_centralizedController_environment import Hexapod_Centralized_Env as HexapodEnv

#from hexapod_envs.hexapod_deploy_default import Hexapod

#from hexapod_envs.phantomX_centralizedController_environment import PhantomX_Centralized_Env as HexapodEnv

ray.init(num_cpus=30, ignore_reinit_error=True)
#ray.init(ignore_reinit_error=True)

config = ppo.DEFAULT_CONFIG.copy()

#config['env'] = "QuantrupedMultiEnv_Centralized"
#config['env'] = "QuantrupedMultiEnv_FullyDecentral"
#config['env'] = "QuantrupedMultiEnv_SingleNeighbor"
#config['env'] = "QuantrupedMultiEnv_SingleDiagonal"
#config['env'] = "QuantrupedMultiEnv_Local"
config['env'] = "AntSix-Centralized" #"PhantomX-v1" #Hexapod
#print("SELECTED ENVIRONMENT: ", policy_scope, " = ", HexapodEnv)

config['num_workers']=2
config['num_envs_per_worker']=4
#config["eager"] = False
#config['nump_gpus']=1

config['train_batch_size'] = 16000#grid_search([4000, 16000, 65536])

# Baseline Defaults:
config['gamma'] = 0.99
config['lambda'] = 0.95 
       
config['entropy_coeff'] = 0. #grid_search([0., 0.01])

config['clip_param'] = 0.2

config['vf_loss_coeff'] = 0.5
#config['vf_clip_param'] = 2500.

config['observation_filter'] = 'MeanStdFilter' #grid_search(['MeanStdFilter', 'NoFilter'])

config['sgd_minibatch_size'] = 128#grid_search([128, 2048]) # BEFORE 2048 # Default: 128, or horizon?
config['num_sgd_iter'] = 10
config['lr'] = 3e-4
config['grad_clip']=0.5

config['model']['custom_model'] = "fc_glorot_uniform_init"
config['model']['fcnet_hiddens'] = [64, 64] #grid_search([ [32, 32],[64, 64],[128, 128]])
#config['model']['vf_share_layers'] = grid_search(['False', ' True'])

#config['seed'] = round(time.time())

# single_env = gym.make("Hexapod-v1")
# #policies = QuantrupedMultiPoliciesEnv.return_policies(single_env.observation_space)
# policies = HexapodEnv.return_policies( spaces.Box(-np.inf, np.inf, (83,), np.float64) )
# 
# config["multiagent"] = {
#         "policies": policies,
#         "policy_mapping_fn": HexapodEnv.policy_mapping_fn,
#         "policies_to_train": HexapodEnv.policy_names, #, "dec_B_policy"],
#     }
# 
config['env_config']['ctrl_cost_weight'] = 0.025 #grid_search([0.01, 0.02, 0.05, 0.1, 0.2])
config['env_config']['contact_cost_weight'] =  0.#0.05 #5e-4#5e-3 #grid_search([5e-4,5e-3,5e-2])
config['env_config']['frame_skip'] = 2 #grid_search([1,2,5])
# config['env_config']['hf_smoothness'] = 1.0
# 
# config['env_config']['curriculum_learning'] =  False
# config['env_config']['range_smoothness'] =  [1., 0.6]
# config['env_config']['range_last_timestep'] =  4000000

# def on_train_result(info):
#     result = info["result"]
#     trainer = info["trainer"]
#     timesteps_res = result["timesteps_total"]
#     trainer.workers.foreach_worker(
#         lambda ev: ev.foreach_env( lambda env: env.update_environment_after_epoch( timesteps_res ) )) 
# 
# config["callbacks"]={"on_train_result": on_train_result,}

analysis = tune.run(
      "PPO",
      name=("Ant6_large_range"),
      num_samples=10,
      checkpoint_at_end=True,
      checkpoint_freq=625,
      stop={"timesteps_total": 50000000},
      config=config,
  )
