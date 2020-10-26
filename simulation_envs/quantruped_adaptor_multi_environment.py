import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces

class QuantrupedMultiPoliciesEnv(MultiAgentEnv):
    """
    """    
    
    policy_names = ["centr_A_policy"]
    
    def __init__(self, config):
        if 'contact_cost_weight' in config.keys():
            contact_cost_weight = config['contact_cost_weight']
        else: 
            contact_cost_weight = 5e-4
        if 'ctrl_cost_weight' in config.keys():
            ctrl_cost_weight = config['ctrl_cost_weight']
        else: 
            ctrl_cost_weight = 0.5
        self.env = gym.make("QuAntruped-v3", 
            ctrl_cost_weight=ctrl_cost_weight,
            contact_cost_weight=contact_cost_weight)
        
        ant_mass = mujoco_py.functions.mj_getTotalmass(self.env.model)
        mujoco_py.functions.mj_setTotalmass(self.env.model, 10. * ant_mass)
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        #self.policy_B = "dec_B_policy"
        
        # From TimeLimit
        #max_episode_steps = 1000
        #if self.env.spec is not None:
         #   self.env.spec.max_episode_steps = max_episode_steps
        #self._max_episode_steps = max_episode_steps
        #self._elapsed_steps = None

    def distribute_observations(self, obs_full):
        return {
            self.policy_names[0]: obs_full,
        }
        
#    def distribute_reward(self, reward_full, info, action_dict):
 #       rew = {}
  #      for policy_name in self.policy_names:
   #         rew[policy_name] = reward_full / len(self.policy_names)
    #    return rew

    def distribute_reward(self, reward_full, info, action_dict):
        fw_reward = info['reward_forward']
        rew = {}      
        for policy_name in self.policy_names:
            rew[policy_name] = fw_reward / len(self.policy_names) - self.env.ctrl_cost_weight * np.sum(np.square(action_dict[policy_name]))
        return rew
        
    def concatenate_actions(self, action_dict):
        return action_dict[self.policy_names[0]]#np.concatenate( (action_dict[self.policy_A],
        
    def reset(self):
        # From TimeLimit
        #self._elapsed_steps = 0
        
        obs_original = self.env.reset()
        return self.distribute_observations(obs_original)

    def step(self, action_dict):
        #assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"    
        obs_full, rew_w, done_w, info_w = self.env.step( self.concatenate_actions(action_dict) ) ##self.env.step( np.concatenate( (action_dict[self.policy_A],
            #action_dict[self.policy_B]) ))
        obs_dict = self.distribute_observations(obs_full)
        
        rew_dict = self.distribute_reward(rew_w, info_w, action_dict)
        
        done = {
            "__all__": done_w,
        }
        
        #self._elapsed_steps += 1
        #if self._elapsed_steps >= self._max_episode_steps:
         #   info_w['TimeLimit.truncated'] = not done
          #  done["__all__"] = True
        
        return obs_dict, rew_dict, done, {}
        
    def render(self):
        self.env.render()
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        return QuantrupedMultiPoliciesEnv.policy_names[0]
            
    @staticmethod
    def return_policies(obs_space):
        policies = {
            QuantrupedMultiPoliciesEnv.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.,-1., -1.,-1.,-1.,-1.]), np.array([+1.,+1.,+1.,+1., +1.,+1.,+1.,+1.])), {}),
        }
        return policies
