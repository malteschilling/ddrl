import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces

from hexapod_envs import HexapodMultiPoliciesEnv
        
class Hexapod_Centralized_Env(HexapodMultiPoliciesEnv):
    """
    """    
    
    # This is ordering of the policies as applied here:
    policy_names = ["central_policy"]
    
    def __init__(self, config):
        self.obs_indices = {}
        self.obs_indices["central_policy"] =  range(0,83)
        super().__init__(config)

    def distribute_observations(self, obs_full):
        """ 
        Construct dictionary that routes to each policy only the relevant
        information.
        """
        obs_distributed = {}
        for policy_name in self.policy_names:
            obs_distributed[policy_name] = obs_full[self.obs_indices[policy_name],]
        return obs_distributed
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        return Hexapod_Centralized_Env.policy_names[0]
            
    @staticmethod
    def return_policies(obs_space):
        policies = {
            Hexapod_Centralized_Env.policy_names[0]: (None,
                obs_space, spaces.Box(-1., 1., (18,) ), {}),
        }
        return policies
