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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--policy_scope", required=False)
args = parser.parse_args()
# Possible values: "QuantrupedMultiEnv_FullyDecentral", "QuantrupedMultiEnv_SingleNeighbor",
# "QuantrupedMultiEnv_SingleDiagonal", "QuantrupedMultiEnv_Local"
if 'policy_scope' in args and args.policy_scope: 
    policy_scope = args.policy_scope
else:
    policy_scope = 'QuantrupedMultiEnv_Centralized'
 
if policy_scope=="QuantrupedMultiEnv_FullyDecentral":
    from simulation_envs.quantruped_adaptor_multi_configurations_environment import QuantrupedFullyDecentralizedEnv as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_SingleNeighbor":
    from simulation_envs.quantruped_adaptor_multi_configurations_environment import Quantruped_LocalSingleNeighboringLeg_Env as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_SingleDiagonal":
    from simulation_envs.quantruped_adaptor_multi_configurations_environment import Quantruped_LocalSingleDiagonalLeg_Env as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_Local":
    from simulation_envs.quantruped_adaptor_multi_configurations_environment import Quantruped_Local_Env as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_TwoSides":
    from simulation_envs.quantruped_adaptor_multi_configurations_environment import Quantruped_TwoSideControllers_Env as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_TwoDiags":
    from simulation_envs.quantruped_adaptor_multi_configurations_environment import Quantruped_TwoDiagControllers_Env as QuantrupedEnv
else:
    from simulation_envs.quantruped_adaptor_multi_environment import QuantrupedMultiPoliciesEnv as QuantrupedEnv


#from simulation_envs.quantruped_adaptor_multi_environment import QuantrupedMultiPoliciesEnv as QuantrupedEnv
#from simulation_envs.quantruped_adaptor_multi_configurations_environment import QuantrupedFullyDecentralizedEnv as QuantrupedEnv
#from simulation_envs.quantruped_adaptor_multi_configurations_environment import Quantruped_LocalSingleNeighboringLeg_Env as QuantrupedEnv
#from simulation_envs.quantruped_adaptor_multi_configurations_environment import Quantruped_LocalSingleDiagonalLeg_Env as QuantrupedEnv
#from simulation_envs.quantruped_adaptor_multi_configurations_environment import Quantruped_Local_Env as QuantrupedEnv

ray.init(num_cpus=15, ignore_reinit_error=True)
#ray.init(ignore_reinit_error=True)

config = ppo.DEFAULT_CONFIG.copy()

#config['env'] = "QuantrupedMultiEnv_Centralized"
#config['env'] = "QuantrupedMultiEnv_FullyDecentral"
#config['env'] = "QuantrupedMultiEnv_SingleNeighbor"
#config['env'] = "QuantrupedMultiEnv_SingleDiagonal"
#config['env'] = "QuantrupedMultiEnv_Local"
config['env'] = policy_scope
print("SELECTED ENVIRONMENT: ", policy_scope, " = ", QuantrupedEnv)

config['num_workers']=2
config['num_envs_per_worker']=4
#config['nump_gpus']=1

config['train_batch_size'] = 4000 #grid_search([4000, 65536]

# Baseline Defaults:
config['gamma'] = 0.99
config['lambda'] = 0.95        
config['entropy_coeff'] = 0.
config['clip_param'] = 0.2

config['vf_loss_coeff'] = 0.5
#config['vf_clip_param'] = 4000.

config['observation_filter'] = 'MeanStdFilter' #grid_search(['MeanStdFilter', 'NoFilter'])

config['sgd_minibatch_size'] = 2048 # Default: 128, or horizon?
config['num_sgd_iter'] = 10
config['lr'] = 3e-4
config['grad_clip']=0.5

config['model']['custom_model'] = "fc_glorot_uniform_init"
config['model']['fcnet_hiddens'] = [64, 64]
#config['model']['vf_share_layers'] = grid_search(['False', ' True'])

#config['seed'] = round(time.time())

single_env = gym.make("QuAntruped-v3")
#policies = QuantrupedMultiPoliciesEnv.return_policies(single_env.observation_space)
policies = QuantrupedEnv.return_policies( spaces.Box(-np.inf, np.inf, (43,), np.float64) )

config["multiagent"] = {
        "policies": policies,
        "policy_mapping_fn": QuantrupedEnv.policy_mapping_fn,
        "policies_to_train": QuantrupedEnv.policy_names, #, "dec_B_policy"],
    }

analysis = tune.run(
      "PPO",
      name=("exp_" + policy_scope),
      num_samples=10,
      checkpoint_at_end=True,
      checkpoint_freq=1042,
      stop={"timesteps_total": 20006400},
      config=config,
  )
