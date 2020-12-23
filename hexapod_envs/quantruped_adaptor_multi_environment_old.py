import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces

class QuantrupedMultiEnv(MultiAgentEnv):
    """
    """    
    def __init__(self, config):
        self.env = gym.make("QuAntruped-v3")
        
        ant_mass = mujoco_py.functions.mj_getTotalmass(self.env.model)
        mujoco_py.functions.mj_setTotalmass(self.env.model, 10. * ant_mass)
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        self.policy_A = "dec_A_policy"
        self.policy_B = "dec_B_policy"
        
        # From TimeLimit
        #max_episode_steps = 1000
        #if self.env.spec is not None:
         #   self.env.spec.max_episode_steps = max_episode_steps
        #self._max_episode_steps = max_episode_steps
        #self._elapsed_steps = None

    def reset(self):
        # From TimeLimit
        #self._elapsed_steps = 0
        
        obs_w = self.env.reset()
        return {
            self.policy_A: obs_w,
            self.policy_B: obs_w,
        }

    def step(self, action_dict):
        #assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"    
        obs_w, rew_w, done_w, info_w = self.env.step(action_dict[self.policy_A]) ##self.env.step( np.concatenate( (action_dict[self.policy_A],
            #action_dict[self.policy_B]) ))
        obs_dict = {
            self.policy_A : obs_w,
            self.policy_B : obs_w,
        }
        
        rew = {
            self.policy_A: 0.5*rew_w,
            self.policy_B: 0.5*rew_w,
        }
        
        done = {
            "__all__": done_w,
        }
        
        #self._elapsed_steps += 1
        #if self._elapsed_steps >= self._max_episode_steps:
         #   info_w['TimeLimit.truncated'] = not done
          #  done["__all__"] = True
        
        return obs_dict, rew, done, {}
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        if agent_id.startswith("dec_A_policy"):
            return "dec_A_policy"
        else:
            return "dec_B_policy" 
            
    @staticmethod
    def return_policies(obs_space):
        policies = {
        "dec_A_policy": (None,
            obs_space, spaces.Box(np.array([-1.,-1.,-1.,-1., -1.,-1.,-1.,-1.]), np.array([+1.,+1.,+1.,+1., +1.,+1.,+1.,+1.])), {}),
        "dec_B_policy": (None,
            obs_space, spaces.Box(np.array([-1.,-1.,-1.,-1.]), np.array([+1.,+1.,+1.,+1.])), {}),
        }
        return policies
