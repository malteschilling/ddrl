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
        self.env = gym.make("QuAntruped-v3")
        
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
        
        rew = {}
        for policy_name in self.policy_names:
            rew[policy_name] = rew_w / len(self.policy_names)
        #rew = {
         #   QuantrupedMultiEnv_Centralized.policy_names[0]: rew_w,
        #}
        
        done = {
            "__all__": done_w,
        }
        
        #self._elapsed_steps += 1
        #if self._elapsed_steps >= self._max_episode_steps:
         #   info_w['TimeLimit.truncated'] = not done
          #  done["__all__"] = True
        
        return obs_dict, rew, done, {}
        
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
