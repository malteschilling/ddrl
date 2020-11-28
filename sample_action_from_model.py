import ray
import numpy as np
import pickle5 as pickle
from ray.tune.registry import get_trainable_cls
import os
from ray.rllib.evaluation.worker_set import WorkerSet

#import simulation_envs
import models
#from rollout_episodes import rollout_episodes

ray_results_dir = os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results_11_02/exp1_20_flat_QuantrupedMultiEnv_Centralized'
       
experiment_dirs = [os.path.join(ray_results_dir,dI) for dI in os.listdir(ray_results_dir) if os.path.isdir(os.path.join(ray_results_dir,dI))]
exp_params = [os.path.join(exp_d, 'params.pkl') for exp_d in experiment_dirs]
exp_checkpoint = [os.path.join(exp_d, 'checkpoint_624', 'checkpoint-624') for exp_d in experiment_dirs]

ray.init()

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

# SAMPLING FROM POLICY
policy = agent.get_policy('centr_A_policy')
test_obs = env.reset()
#policy.compute_actions(test_obs['centr_A_policy'].reshape(1,43))

test_dict = {}
test_dict[policy._obs_input] = test_obs['centr_A_policy'].reshape(1,43)
#policy._sess.run(builder.fetches, test_dict)
#to_fetch = [self._sampled_action] + self._state_outputs + \
 #                  [self.extra_compute_action_fetches()]
test_fetches = [policy._sampled_action] + [policy.extra_compute_action_fetches()]
f_test = policy._sess.run(test_fetches, test_dict)
print(f_test)