import ray
import pickle5 as pickle
import os
import collections

from ray.tune.registry import get_trainable_cls
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.env.base_env import _DUMMY_AGENT_ID

import simulation_envs
import gym

class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""
    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value

def keep_going(steps, num_steps, episodes, num_episodes):
    """Determine whether we've collected enough data"""
    # if num_episodes is set, this overrides num_steps
    if num_episodes:
        return episodes < num_episodes
    # if num_steps is set, continue until we reach the limit
    if num_steps:
        return steps < num_steps
    # otherwise keep going forever
    return True

def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID

config_checkpoint="/Users/mschilling/Desktop/develop/Decentralized_DRL/ray_results/rllib_centralized_2/PPO_QuantrupedMultiEnv_Centralized_7443a_00000_0_2020-10-21_20-23-02/checkpoint_3125/checkpoint-3125"
config_checkpoint="/Users/mschilling/ray_results/rllib_quantruped/PPO_QuAntruped-v3_e38c0_00000_0_2020-10-14_11-51-53/checkpoint_625/checkpoint-625"
config_dir = os.path.dirname(config_checkpoint)
config_path = os.path.join(config_dir, "params.pkl")

# Try parent directory.
if not os.path.exists(config_path):
    config_path = os.path.join(config_dir, "../params.pkl")
    
if os.path.exists(config_path):
    with open(config_path, "rb") as f:
        config = pickle.load(f)
        
if "num_workers" in config:
    config["num_workers"] = min(2, config["num_workers"])
    
ray.init()

cls = get_trainable_cls('PPO')
agent = cls(env=config['env'], config=config)
# Load state from checkpoint.
agent.restore(config_checkpoint)
num_steps = int(1000)
num_episodes = int(3)

env = agent.workers.local_worker().env
#gym.make("QuAntruped-v3")
multiagent = False

try:
    policy_map = {DEFAULT_POLICY_ID: agent.get_policy()}
except AttributeError:
    raise AttributeError(
        "Agent ({}) does not have a `policy` property! This is needed "
        "for performing (trained) agent rollouts.".format(agent))

use_lstm = {DEFAULT_POLICY_ID: False}

action_init = {
    p: flatten_to_single_ndarray(m.action_space.sample())
    for p, m in policy_map.items()
}

steps = 0
episodes = 0

no_render = False
policy_agent_mapping = default_policy_agent_mapping

while keep_going(steps, num_steps, episodes, num_episodes):
    mapping_cache = {}  # in case policy_agent_mapping is stochastic
#    saver.begin_rollout()
    obs = env.reset()
    agent_states = DefaultMapping(
        lambda agent_id: state_init[mapping_cache[agent_id]])
    prev_actions = DefaultMapping(
        lambda agent_id: action_init[mapping_cache[agent_id]])
    prev_rewards = collections.defaultdict(lambda: 0.)
    done = False
    reward_total = 0.0
    while not done and keep_going(steps, num_steps, episodes,
                                  num_episodes):
        multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
        action_dict = {}
        for agent_id, a_obs in multi_obs.items():
            if a_obs is not None:
                policy_id = mapping_cache.setdefault(
                    agent_id, policy_agent_mapping(agent_id))
                p_use_lstm = use_lstm[policy_id]
                if p_use_lstm:
                    a_action, p_state, _ = agent.compute_action(
                        a_obs,
                        state=agent_states[agent_id],
                        prev_action=prev_actions[agent_id],
                        prev_reward=prev_rewards[agent_id],
                        policy_id=policy_id)
                    agent_states[agent_id] = p_state
                else:
                    a_action = agent.compute_action(
                        a_obs,
                        prev_action=prev_actions[agent_id],
                        prev_reward=prev_rewards[agent_id],
                        policy_id=policy_id)
                a_action = flatten_to_single_ndarray(a_action)
                action_dict[agent_id] = a_action
                prev_actions[agent_id] = a_action
        action = action_dict
        action = action if multiagent else action[_DUMMY_AGENT_ID]
        next_obs, reward, done, info = env.step(action)
        if multiagent:
            for agent_id, r in reward.items():
                prev_rewards[agent_id] = r
        else:
            prev_rewards[_DUMMY_AGENT_ID] = reward
        if multiagent:
            done = done["__all__"]
            reward_total += sum(reward.values())
        else:
            reward_total += reward
        if not no_render:
            env.render()
#        saver.append_step(obs, action, next_obs, reward, done, info)
        steps += 1
        obs = next_obs
#    saver.end_rollout()
    print("Episode #{}: reward: {}".format(episodes, reward_total))
    if done:
        episodes += 1