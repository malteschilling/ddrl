import ray
import pickle5 as pickle
import os
import numpy as np
from numpy.ma import masked_array
import numpy.ma as ma

import matplotlib.pyplot as plt
from matplotlib import cm

from ray.tune.registry import get_trainable_cls
from ray.rllib.evaluation.worker_set import WorkerSet

import simulation_envs
import models

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

agent.restore(exp_checkpoint[0])

if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
    env = agent.workers.local_worker().env
    
policy = agent.get_policy('centr_A_policy')

test_obs = env.reset()

test_dict = {}
test_dict[policy._obs_input] = test_obs['centr_A_policy'].reshape(1,43)
test_fetches = [policy._sampled_action] + [policy.extra_compute_action_fetches()]
f_test = policy._sess.run(test_fetches, test_dict)

# Computing over actions:
weight_paths = np.zeros((43,8))
for i in range(0,43):
    input = np.zeros( (1000,43) )
    input[:,i] = 1.
    output_actions = policy.compute_actions( input )[0]
    weight_paths[i,:] = np.mean(output_actions, axis=0)
    
weight_path_abs = np.abs(weight_paths)
weight_path_abs = weight_path_abs/np.max(weight_path_abs)
weight_path_abs_norm = weight_path_abs / weight_path_abs.sum(axis=0)
weight_path_abs_sum_norm = weight_path_abs / weight_path_abs.sum(axis=0)

wpath_centr = ma.array(weight_path_abs_norm)
wpath_fl = ma.array(weight_path_abs_norm)
wpath_hl = ma.array(weight_path_abs_norm)
wpath_fr = ma.array(weight_path_abs_norm)
wpath_hr = ma.array(weight_path_abs_norm)

# Using both, negative and positive activation

weight_paths_neg = np.zeros((43,8))
for i in range(0,43):
    input = np.zeros( (1000,43) )
    input[:,i] = -1.
    output_actions = policy.compute_actions( input )[0]
    weight_paths_neg[i,:] = np.mean(output_actions, axis=0)
    
weight_path_neg_abs = np.abs(weight_paths_neg)
weight_path_neg_abs = weight_path_neg_abs/np.max(weight_path_neg_abs)

weight_path_abs_sum = weight_path_neg_abs + weight_path_abs
weight_path_abs_sum_norm = weight_path_abs_sum / weight_path_abs_sum.sum(axis=0)

wpath_centr = ma.array(weight_path_abs_sum_norm)
wpath_fl = ma.array(weight_path_abs_sum_norm)
wpath_hl = ma.array(weight_path_abs_sum_norm)
wpath_fr = ma.array(weight_path_abs_sum_norm)
wpath_hr = ma.array(weight_path_abs_sum_norm)

wpath_centr[5:13,:] = ma.masked
wpath_centr[19:43,:] = ma.masked

wpath_fl[0:5,:] = ma.masked
wpath_fl[7:19,:] = ma.masked
wpath_fl[21:27,:] = ma.masked
wpath_fl[29:37,:] = ma.masked
wpath_fl[39:43,:] = ma.masked

wpath_hl[0:7,:] = ma.masked
wpath_hl[9:21,:] = ma.masked
wpath_hl[23:29,:] = ma.masked
wpath_hl[31:39,:] = ma.masked
wpath_hl[41:43,:] = ma.masked

wpath_hr[0:9,:] = ma.masked
wpath_hr[11:23,:] = ma.masked
wpath_hr[25:31,:] = ma.masked
wpath_hr[33:41,:] = ma.masked

wpath_fr[0:11,:] = ma.masked
wpath_fr[13:25,:] = ma.masked
wpath_fr[27:33,:] = ma.masked
wpath_fr[37:43,:] = ma.masked

fig, ax = plt.subplots(figsize=(6, 8))
#ax.set_rasterized(True)

(ax.imshow(wpath_centr, interpolation='none', cmap=cm.Greys)).set_rasterized(True)
(ax.imshow(wpath_fl, interpolation='none', cmap=cm.Purples)).set_rasterized(True)
(ax.imshow(wpath_hl, interpolation='none', cmap=cm.Blues)).set_rasterized(True)
(ax.imshow(wpath_hr, interpolation='none', cmap=cm.Greens)).set_rasterized(True)
(ax.imshow(wpath_fr, interpolation='none', cmap=cm.Oranges)).set_rasterized(True)
ax.set_aspect('auto')

ax.set_xticks(np.arange(8))
ax.set_xticklabels(["FR", "FR knee", "FL", "FL knee", "HL", "HL knee", "HR", "HR knee"])

ax.get_xticklabels()[0].set_color("orange")
ax.get_xticklabels()[1].set_color("orange")
ax.get_xticklabels()[2].set_color("purple")
ax.get_xticklabels()[3].set_color("purple")
ax.get_xticklabels()[4].set_color("blue")
ax.get_xticklabels()[5].set_color("blue")
ax.get_xticklabels()[6].set_color("green")
ax.get_xticklabels()[7].set_color("green")

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

plt.show()