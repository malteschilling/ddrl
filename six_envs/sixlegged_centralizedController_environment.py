import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces

from six_envs import SixLeggedMultiPoliciesEnv
        
class SixLegged_Centralized_Env(SixLeggedMultiPoliciesEnv):
    """
    """    
    
    # This is ordering of the policies as applied here:
    policy_names = ["central_policy"]
    
    def __init__(self, config):
        self.obs_indices = {}
        # First global information: 
        # 0: height, 1-4: quaternion orientation torso
        #  5,  6,  7: FL joint angles
        #  8,  9, 10: FR joint angles
        # 11, 12, 13: ML joint angles
        # 14, 15, 16: MR joint angles
        # 17, 18, 19: HL joint angles
        # 20, 21, 22: HR joint angles
        # Velocities follow same ordering, but have in addition x and y vel.
        # 23-25: vel, 26-28: rotational velocity
        # 29, 30, 31: FL joint velocity
        # 32, 33, 34: FR joint velocity
        # 35, 36, 37: ML joint velocity
        # 38, 39, 40: MR joint velocity
        # 41, 42, 43: HL joint velocity
        # 44, 45, 46: HR joint velocity
        # Passive forces same ordering, only local information used
        # 47, 48, 49: FL passive torque
        # 50, 51, 52: FR passive torque
        # 53, 54, 55: ML passive torque
        # 56, 57, 58: MR passive torque
        # 59, 60, 61: HL passive torque
        # 62, 63, 64: HR passive torque
        # Last: control signals (clipped) from last time step
        # 65, 66, 67: FL ctrl
        # 68, 69, 70: FR ctrl
        # 71, 72, 73: ML ctrl
        # 74, 75, 76: MR ctrl
        # 77, 78, 79: HL ctrl
        # 80, 81, 82: HR ctrl
        # The central policy gets all observations
        self.obs_indices["central_policy"] =  range(0,83)
        super().__init__(config)

    def distribute_observations(self, obs_full):
        """ 
        Construct dictionary that routes to each policy only the relevant
        information.
        """
        obs_distributed = {}
        for policy_name in self.policy_names:
            obs_distributed[policy_name] = obs_full[self.obs_indices[policy_name],].copy()
        return obs_distributed
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        return SixLegged_Centralized_Env.policy_names[0]
            
    @staticmethod
    def return_policies():
        policies = {
            SixLegged_Centralized_Env.policy_names[0]: (None,
                spaces.Box(-np.inf, np.inf, (83,), np.float64),
                spaces.Box(-1., +1., (18,)), {}),
        }
        return policies
