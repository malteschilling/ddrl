import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces

from hexapod_envs.hexapod_adaptor_multi_environment import HexapodMultiPoliciesEnv

class HexapodSixControllerSuperEnv(HexapodMultiPoliciesEnv):
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
        contact_cost[self.policy_names[0]] = global_contact_costs + np.sum(contact_costs[2:5])
        contact_cost[self.policy_names[1]] = global_contact_costs + np.sum(contact_costs[5:8])
        contact_cost[self.policy_names[2]] = global_contact_costs + np.sum(contact_costs[8:11])
        contact_cost[self.policy_names[3]] = global_contact_costs + np.sum(contact_costs[11:14])
        contact_cost[self.policy_names[4]] = global_contact_costs + np.sum(contact_costs[14:17])
        contact_cost[self.policy_names[5]] = global_contact_costs + np.sum(contact_costs[17:])
        #print(contact_cost)
        #sum_c = 0.
        #for i in self.policy_names:
         #   sum_c += contact_cost[i]
        #print("Calculated: ", sum_c)
        return contact_cost
        
    def concatenate_actions(self, action_dict):
        # Return actions
        actions = np.concatenate( (action_dict[self.policy_names[0]],
            action_dict[self.policy_names[1]],
            action_dict[self.policy_names[2]],
            action_dict[self.policy_names[3]],
            action_dict[self.policy_names[4]],
            action_dict[self.policy_names[5]]) )
        return actions
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        if agent_id.startswith("policy_FL"):
            return "policy_FL"
        elif agent_id.startswith("policy_FR"):
            return "policy_FR"
        elif agent_id.startswith("policy_ML"):
            return "policy_ML"
        elif agent_id.startswith("policy_MR"):
            return "policy_MR"
        elif agent_id.startswith("policy_HL"):
            return "policy_HL"
        else:
            return "policy_HR" 


class HexapodFullyDecentralizedEnv(HexapodSixControllerSuperEnv):
    """
    """    
    
    # This is ordering of the policies as applied here:
    policy_names = ["policy_FL", "policy_FR",
        "policy_ML","policy_MR",
        "policy_HL","policy_HR"]
    
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
        # 83: target_velocity
        self.obs_indices["policy_FL"] = [0,1,2,3,4, 5, 6, 7,23,24,25,26,27,28,29,30,31,47,48,49,65,66,67,83]
        self.obs_indices["policy_FR"] = [0,1,2,3,4, 8, 9,10,23,24,25,26,27,28,32,33,34,50,51,52,68,69,70,83]
        self.obs_indices["policy_ML"] = [0,1,2,3,4,11,12,13,23,24,25,26,27,28,35,36,37,53,54,55,71,72,73,83]
        self.obs_indices["policy_MR"] = [0,1,2,3,4,14,15,16,23,24,25,26,27,28,38,39,40,56,57,58,74,75,76,83]
        self.obs_indices["policy_HL"] = [0,1,2,3,4,17,18,19,23,24,25,26,27,28,41,42,43,59,60,61,77,78,79,83]
        self.obs_indices["policy_HR"] = [0,1,2,3,4,20,21,22,23,24,25,26,27,28,44,45,46,62,63,64,80,81,82,83]
        super().__init__(config)
            
    @staticmethod
    def return_policies():
        obs_space = spaces.Box(-np.inf, np.inf, (24,), np.float64)
        policies = {
            HexapodFullyDecentralizedEnv.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1]), np.array([+1.,+1.,+1])), {}),
            HexapodFullyDecentralizedEnv.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1])), {}),
            HexapodFullyDecentralizedEnv.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
            HexapodFullyDecentralizedEnv.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1])), {}),
            HexapodFullyDecentralizedEnv.policy_names[4]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1]), np.array([+1.,+1.,+1])), {}),
            HexapodFullyDecentralizedEnv.policy_names[5]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1]), np.array([+1.,+1.,+1])), {}),
        }
        return policies
        
class Hexapod_Local_Env(HexapodSixControllerSuperEnv):
    """
    """    
    
    # This is ordering of the policies as applied here:
    policy_names = policy_names = ["policy_FL", "policy_FR",
        "policy_ML","policy_MR",
        "policy_HL","policy_HR"]
    
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
        # 83: target_velocity
        self.obs_indices["policy_FL"] = [0,1,2,3,4, 
            5, 6, 7, 8, 9,10,11,12,13,
            23,24,25,26,27,28,
            29,30,31,32,33,34,35,36,37,
            47,48,49,50,51,52,53,54,55,
            65,66,67,68,69,70,71,72,73,
            83]
        self.obs_indices["policy_FR"] = [0,1,2,3,4, 
            8, 9,10,5, 6, 7,14,15,16,
            23,24,25,26,27,28,
            32,33,34,29,30,31,38,39,40,
            50,51,52,47,48,49,56,57,58,
            68,69,70,65,66,67,74,75,76,
            83]
        self.obs_indices["policy_ML"] = [0,1,2,3,4,
            11,12,13,5, 6, 7,17,18,19,
            23,24,25,26,27,28,
            35,36,37,29,30,31,41,42,43,
            53,54,55,47,48,49,59,60,61,
            71,72,73,65,66,67,77,78,79,
            83]
        self.obs_indices["policy_MR"] = [0,1,2,3,4,
            14,15,16,8, 9,10,20,21,22,
            23,24,25,26,27,28,
            38,39,40,32,33,34,44,45,46,
            56,57,58,50,51,52,62,63,64,
            74,75,76,68,69,70,80,81,82,
            83]
        self.obs_indices["policy_HL"] = [0,1,2,3,4,
            17,18,19,11,12,13,20,21,22,
            23,24,25,26,27,28,
            41,42,43,35,36,37,44,45,46,
            59,60,61,53,54,55,62,63,64,
            77,78,79,71,72,73,80,81,82,
            83]
        self.obs_indices["policy_HR"] = [0,1,2,3,4,
            20,21,22,17,18,19,14,15,16,
            23,24,25,26,27,28,
            44,45,46,41,42,43,38,39,40,
            62,63,64,59,60,61,56,57,58,
            80,81,82,77,78,79,74,75,76,
            83]
        
        super().__init__(config)
        
#    def distribute_reward(self, reward_full, info, action_dict):
 #       fw_reward = info['reward_forward']
  #      rew = {}      
   #     for policy_name in self.policy_names:
    #        rew[policy_name] = fw_reward / len(self.policy_names) - self.env.ctrl_cost_weight * np.sum(np.square(action_dict[policy_name]))
     #   return rew
            
    @staticmethod
    def return_policies():
        obs_space = spaces.Box(-np.inf, np.inf, (48,), np.float64)
        policies = {
            Hexapod_Local_Env.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
            Hexapod_Local_Env.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
            Hexapod_Local_Env.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
            Hexapod_Local_Env.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
            Hexapod_Local_Env.policy_names[4]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
            Hexapod_Local_Env.policy_names[5]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
        }
        return policies
