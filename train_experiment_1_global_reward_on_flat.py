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

from ray.rllib.agents.callbacks import DefaultCallbacks

import exp1_simulation_envs
import models

import argparse

# Switch between different approaches.
parser = argparse.ArgumentParser()
parser.add_argument("--policy_scope", required=False)
args = parser.parse_args()
# Possible values: 
#   QuantrupedMultiEnv_Centralized - single controller, global information
#   QuantrupedMultiEnv_FullyDecentral - four decentralized controlller, information 
#       from the controlled leg only
#   QuantrupedMultiEnv_SingleNeighbor - four decentralized controlller, information 
#       from the controlled leg plus neighbor (ccw)
#   QuantrupedMultiEnv_SingleDiagonal - four decentralized controlller, information 
#       from the controlled leg plus diagonal
#   QuantrupedMultiEnv_SingleToFront - four decentralized controlller, information 
#       from the controlled leg plus one neighbor, for front legs from hind legs
#       for hind legs, the other hind leg
#   QuantrupedMultiEnv_Local - four decentralized controlller, information 
#       from the controlled leg plus both neighboring legs
#   QuantrupedMultiEnv_TwoSides - two decentralized controlller, one for each side, 
#       information from the controlled legs 
#   QuantrupedMultiEnv_TwoDiags - two decentralized controlller, controlling a pair of 
#       diagonal legs, 
#       information from the controlled legs 
#   QuantrupedMultiEnv_FullyDecentralGlobalCost - four decentralized controlller, information 
#       from the controlled leg; variation: global costs are used.

if 'policy_scope' in args and args.policy_scope: 
    policy_scope = args.policy_scope
else:
    policy_scope = 'QuantrupedMultiEnv_Centralized'
 
if policy_scope=="QuantrupedMultiEnv_FullyDecentral":
    from exp1_simulation_envs.quantruped_fourDecentralizedController_environments import QuantrupedFullyDecentralizedEnv as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_SingleNeighbor":
    from exp1_simulation_envs.quantruped_fourDecentralizedController_environments import Quantruped_LocalSingleNeighboringLeg_Env as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_SingleDiagonal":
    from exp1_simulation_envs.quantruped_fourDecentralizedController_environments import Quantruped_LocalSingleDiagonalLeg_Env as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_SingleToFront":
    from exp1_simulation_envs.quantruped_fourDecentralizedController_environments import Quantruped_LocalSingleToFront_Env as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_Local":
    from exp1_simulation_envs.quantruped_fourDecentralizedController_environments import Quantruped_Local_Env as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_TwoSides":
    from exp1_simulation_envs.quantruped_twoDecentralizedController_environments import Quantruped_TwoSideControllers_Env as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_TwoDiags":
    from exp1_simulation_envs.quantruped_twoDecentralizedController_environments import Quantruped_TwoDiagControllers_Env as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_FullyDecentralGlobalCost":
    from exp1_simulation_envs.quantruped_fourDecentralizedController_GlobalCosts_environments import QuantrupedFullyDecentralizedGlobalCostEnv as QuantrupedEnv
else:
    from exp1_simulation_envs.quantruped_centralizedController_environment import Quantruped_Centralized_Env as QuantrupedEnv

# Init ray: First line on server, second for laptop
#ray.init(num_cpus=30, ignore_reinit_error=True)
ray.init(ignore_reinit_error=True)

config = ppo.DEFAULT_CONFIG.copy()

config['env'] = policy_scope
print("SELECTED ENVIRONMENT: ", policy_scope, " = ", QuantrupedEnv)

config['num_workers']=2
config['num_envs_per_worker']=4
#config['nump_gpus']=1

# used grid_search([4000, 16000, 65536], didn't matter too much
config['train_batch_size'] = 16000 

# Baseline Defaults:
config['gamma'] = 0.99
config['lambda'] = 0.95 
       
config['entropy_coeff'] = 0. # again used grid_search([0., 0.01]) for diff. values from lit.
config['clip_param'] = 0.2

config['vf_loss_coeff'] = 0.5
#config['vf_clip_param'] = 4000.

config['observation_filter'] = 'MeanStdFilter'

config['sgd_minibatch_size'] = 128
config['num_sgd_iter'] = 10
config['lr'] = 3e-4
config['grad_clip']=0.5

config['model']['custom_model'] = "fc_glorot_uniform_init"
config['model']['fcnet_hiddens'] = [64, 64]

#config['seed'] = round(time.time())

# For running tune, we have to provide information on 
# the multiagent which are part of the MultiEnvs
policies = QuantrupedEnv.return_policies( spaces.Box(-np.inf, np.inf, (43,), np.float64) )

config["multiagent"] = {
        "policies": policies,
        "policy_mapping_fn": QuantrupedEnv.policy_mapping_fn,
        "policies_to_train": QuantrupedEnv.policy_names, #, "dec_B_policy"],
    }

config['env_config']['ctrl_cost_weight'] = 0.5#grid_search([5e-4,5e-3,5e-2])
config['env_config']['contact_cost_weight'] =  5e-2 #grid_search([5e-4,5e-3,5e-2])

# Parameters for defining environment:
# Heightfield smoothness (between 0.6 and 1.0 are OK)
config['env_config']['hf_smoothness'] = 1.0
# Defining curriculum learning
config['env_config']['curriculum_learning'] =  False
config['env_config']['range_smoothness'] =  [1., 0.6]
config['env_config']['range_last_timestep'] =  10000000

# For curriculum learning: environment has to be updated every epoch
# def on_train_result(info):
#     result = info["result"]
#     trainer = info["trainer"]
#     timesteps_res = result["timesteps_total"]
#     trainer.workers.foreach_worker(
#         lambda ev: ev.foreach_env( lambda env: env.update_environment_after_epoch( timesteps_res ) )) 
# config["callbacks"]={"on_train_result": on_train_result,}
class curriculumCallback(DefaultCallbacks):
    def on_train_result(self, *, trainer, result: dict, **kwargs):
        timesteps_res = result["timesteps_total"]
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(lambda env: env.update_environment_after_epoch(timesteps_res)))
config["callbacks"] = curriculumCallback  # {"on_train_result": on_train_result, }



# Call tune and run (for evaluation: 10 seeds up to 20M steps; only centralized controller
# required that much of time; decentralized controller should show very good results 
# after 5M steps.
analysis = tune.run(
      "PPO",
      name=("GR_" + policy_scope),
      num_samples=10,
      checkpoint_at_end=True,
      checkpoint_freq=312,
      stop={"timesteps_total": 20000000},
      config=config,
  )
