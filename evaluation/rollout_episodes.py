import collections

from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.env import MultiAgentEnv

import numpy as np
import mujoco_py
from mujoco_py import functions

import matplotlib.pyplot as plt

"""
    Running a learned (multiagent) controller,
    for evaluation or visualisation.
    This is adapted from rllib's rollout.py
    (github.com/ray/rllib/rollout.py)
"""

class DefaultMapping(collections.defaultdict):
    """ Provides a default mapping.
    """
    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value

def rollout_episodes(env, agent, num_episodes=1, num_steps=1000, render=True, save_images=None, explore_during_rollout=None, tvel=None, save_obs=None):
    """
        Rollout an episode:
        step through an episode, using the 
        - agent = trained policies (is a multiagent consisting of a dict of agents)
        - env = in the given environment
        for num_steps control steps and running num_episodes episodes.
        
        render: shows OpenGL window
        save_images: save individual frames (can be combined to video)
        tvel: set target velocity
    """
    if tvel:
        env.target_velocity_list = [tvel]
    # Setting up the agent for running an episode.
    multiagent = isinstance(env, MultiAgentEnv)
    if agent.workers.local_worker().multiagent:
        policy_agent_mapping = agent.config["multiagent"]["policy_mapping_fn"]
    policy_map = agent.workers.local_worker().policy_map
    state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
    use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    
    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }
    
    #if save_images:
     #   viewer = mujoco_py.MjRenderContextOffscreen(env.env.sim, 0)
    
    # Collecting statistics over episodes.
    reward_eps = []
    cot_eps = []
    vel_eps = []
    dist_eps = []
    steps_eps = []
    power_total_eps = []
    if save_obs:
        obs_list = []
    for episodes in range(0, num_episodes):
        # Reset all values for this episode.
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        #    saver.begin_rollout()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        power_total = 0.0
        steps = 0
        done = False
        env.env.create_new_random_hfield()
        obs = env.reset()
        start_pos = env.env.sim.data.qpos[0]
        # Control stepping:
        while not done and steps<num_steps:
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
                            policy_id=policy_id,
                            explore=explore_during_rollout)
                        agent_states[agent_id] = p_state
                    else:
                        # Sample an action for the current observation 
                        # for one entry of the agent dict.
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
            # Stepping the environment.
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
            if render:
                if save_images:
                    #viewer.render(1280, 800, 0)
                    if tvel:
                        env.env.model.body_pos[14][0] += tvel * 0.05
                    img = env.env.sim.render(width=1280,height=800, camera_name="side_run")
                    #data = np.asarray(viewer.read_pixels(800, 1280, depth=False)[::-1, :, :], dtype=np.uint8)                
                    #img_array = env.env.render('rgb_array')
                    plt.imsave(save_images + str(steps).zfill(4) + '.png', img, origin='lower')
                else:
                    env.render()
            #saver.append_step(obs, action, next_obs, reward, done, info)
            steps += 1
            obs = next_obs
            if save_obs:
                obs_list.append(obs)
            # Calculated as torque (during last time step - or in this case sum of 
            # proportional control signal (clipped to [-1,1], multiplied by 150 to torque)
            # multiplied by joint velocity for each joint.
            # Important: unfortunately there is a shift in the ctrl signals - therefore use roll
            # (control signals start with front right leg, front left leg starts at index 2)
            current_power = np.sum(np.abs(np.roll(env.env.sim.data.ctrl, -2) * env.env.sim.data.qvel[6:]))
            power_total += current_power
        #saver.end_rollout()
        distance_x = env.env.sim.data.qpos[0] - start_pos
        com_vel = distance_x/steps
        cost_of_transport = (power_total/steps) / (mujoco_py.functions.mj_getTotalmass(env.env.model) * com_vel)
        # Weight is 8.78710174560547
        #print(steps, " - ", power_total, " / ", power_total/steps, "; CoT: ", cost_of_transport)
        cot_eps.append(cost_of_transport)
        reward_eps.append(reward_total)
        vel_eps.append(com_vel)
        dist_eps.append(distance_x)
        steps_eps.append(steps)
        power_total_eps.append(power_total)
        #print(episodes, ' - ', reward_total, '; CoT: ', cost_of_transport, '; Vel: ', com_vel)
    # Return collected information from episode.
    if save_obs:
        np.save( str(save_obs+'/obs_list'), obs_list)
    return (reward_eps, steps_eps, dist_eps, power_total_eps, vel_eps, cot_eps )