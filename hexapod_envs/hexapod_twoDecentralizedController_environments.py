import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces

from hexapod_envs import HexapodMultiPoliciesEnv
        
class Hexapod_TwoSideControllers_Env(HexapodMultiPoliciesEnv):
    """
    """    
    
    # This is ordering of the policies as applied here:
    policy_names = ["policy_LEFT","policy_RIGHT"]
    
    def __init__(self, config):
        self.obs_indices = {}
        # First global information: 
        # 0: height, 1-4: quaternion orientation torso
        # 5 - 7: FL, 8 - 10: FR, 11-13: ML, 14-16: MR, 17-19: HL, 20-22: HR
        # Velocities follow same ordering, but have in addition x and y vel.
        # 23-25: vel, 26-28: rotational velocity
        # 29-31: FL, 32-34: FR, 35-37: ML, 38-40: MR, 41-43: HL, 44-46: HR
        # Passive forces same ordering, only local information used
        # 47-49: FL, 50-52: FR, 53-55: ML, 56-58: MR, 59-61: HL, 62-64: HR
        # Last: control signals (clipped) from last time step
        # 65-67: FL, 68-70: FR, 71-73: ML, 74-76: MR, 77-79: HL, 80-82: HR
        self.obs_indices["policy_LEFT"] =  [0,1,2,3,4, 
            5, 6, 7,
            23,24,25,26,27,28,
            29,30,31,
            47,48,49,
            65,66,67,
            11,12,13,
            35,36,37,
            53,54,55,
            71,72,73,
            17,18,19,
            41,42,43,
            59,60,61,
            77,78,79]
        # Each controller only gets information from that body side: Right
        self.obs_indices["policy_RIGHT"] = [0,1,2,3,4, 
            8, 9,10,
            23,24,25,26,27,28,
            32,33,34,
            50,51,52,
            68,69,70,
            14,15,16,
            38,39,40,
            56,57,58,
            74,75,76,
            20,21,22,
            44,45,46,
            62,63,64,
            80,81,82]
        super().__init__(config)

    def distribute_observations(self, obs_full):
        """ 
        Construct dictionary that routes to each policy only the relevant
        local information.
        """
        obs_distributed = {}
        for policy_name in self.policy_names:
            obs_distributed[policy_name] = obs_full[self.obs_indices[policy_name],]
        return obs_distributed

    def distribute_contact_cost(self):
        contact_cost = {}
        #print("CONTACT COST")
        #from mujoco_py import functions
        #functions.mj_rnePostConstraint(self.env.model, self.env.data)
        #print("From Ant Env: ", self.env.contact_cost)
        raw_contact_forces = self.env.sim.data.cfrc_ext
        contact_forces = np.clip(raw_contact_forces, -1., 1.)
        contact_costs = self.env.contact_cost_weight * np.square(contact_forces)
        global_contact_costs = np.sum(contact_costs[0:2])/4.
        contact_cost[self.policy_names[0]] = global_contact_costs + np.sum(contact_costs[2:5]) + np.sum(contact_costs[8:11]) + np.sum(contact_costs[14:17])
        contact_cost[self.policy_names[1]] = global_contact_costs + np.sum(contact_costs[5:8]) + np.sum(contact_costs[11:14]) + np.sum(contact_costs[17:])
        #print(contact_cost)
        #sum_c = 0.
        #for i in self.policy_names:
         #   sum_c += contact_cost[i]
        #print("Calculated: ", sum_c)
        return contact_cost
        
    def concatenate_actions(self, action_dict):
        # Return actions in the (DIFFERENT in Mujoco) order FR - FL - HL - HR
        actions = np.concatenate( (action_dict[self.policy_names[0]][0:3],
            action_dict[self.policy_names[1]][0:3],
            action_dict[self.policy_names[0]][3:6],
            action_dict[self.policy_names[1]][3:6],
            action_dict[self.policy_names[0]][6:],
            action_dict[self.policy_names[1]][6:]) )
        return actions
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        if agent_id.startswith("policy_LEFT"):
            return "policy_LEFT"
        else:
            return "policy_RIGHT" 
            
    @staticmethod
    def return_policies(obs_space):
        obs_space = spaces.Box(-np.inf, np.inf, (47,), np.float64)
        policies = {
            Hexapod_TwoSideControllers_Env.policy_names[0]: (None,
                obs_space, spaces.Box(-1., 1., (9,) ), {}),
            Hexapod_TwoSideControllers_Env.policy_names[1]: (None,
                obs_space, spaces.Box(-1., 1., (9,) ), {}),
        }
        return policies
    