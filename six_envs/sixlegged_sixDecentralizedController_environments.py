import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces

from six_envs import SixLeggedMultiPoliciesEnv

class SixLeggedControllerSuperEnv(SixLeggedMultiPoliciesEnv):
    def distribute_observations(self, obs_full):
        """ 
        Construct dictionary that routes to each policy only the relevant
        local information.
        """
        obs_distributed = {}
        for policy_name in self.policy_names:
            obs_distributed[policy_name] = obs_full[self.obs_indices[policy_name],].copy()
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
        global_contact_costs = np.sum(contact_costs[0:2])/6.
        contact_cost[self.policy_names[0]] = global_contact_costs + np.sum(contact_costs[2:6])
        contact_cost[self.policy_names[1]] = global_contact_costs + np.sum(contact_costs[6:10])
        contact_cost[self.policy_names[2]] = global_contact_costs + np.sum(contact_costs[10:14])
        contact_cost[self.policy_names[3]] = global_contact_costs + np.sum(contact_costs[14:18])
        contact_cost[self.policy_names[4]] = global_contact_costs + np.sum(contact_costs[18:22])
        contact_cost[self.policy_names[5]] = global_contact_costs + np.sum(contact_costs[22:])
        #print(contact_cost)
        #sum_c = 0.
        #for i in self.policy_names:
         #   sum_c += contact_cost[i]
        #print("Calculated: ", sum_c)
        return contact_cost
        
    def concatenate_actions(self, action_dict):
        # Return actions: FL, FR, ML, MR, HL, HR
        actions = np.concatenate( (action_dict[self.policy_names[0]],
            action_dict[self.policy_names[1]],
            action_dict[self.policy_names[2]],
            action_dict[self.policy_names[3]],
            action_dict[self.policy_names[4]],
            action_dict[self.policy_names[5]],) )
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


class SixLeggedFullyDecentralizedEnv(SixLeggedControllerSuperEnv):
    """
    """    
    
    # This is ordering of the policies as applied here:
    policy_names = ["policy_FL","policy_FR",
        "policy_ML","policy_MR",
        "policy_HL","policy_HR",]
    
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
        # Each leg gets only observations from that particular leg.
        self.obs_indices["policy_FL"] = [0,1,2,3,4, 5, 6, 7,23,24,25,26,27,28,29,30,31,47,48,49,65,66,67]
        self.obs_indices["policy_FR"] = [0,1,2,3,4, 8, 9,10,23,24,25,26,27,28,32,33,34,50,51,52,68,69,70]
        self.obs_indices["policy_ML"] = [0,1,2,3,4,11,12,13,23,24,25,26,27,28,35,36,37,53,54,55,71,72,73]
        self.obs_indices["policy_MR"] = [0,1,2,3,4,14,15,16,23,24,25,26,27,28,38,39,40,56,57,58,74,75,76]
        self.obs_indices["policy_HL"] = [0,1,2,3,4,17,18,19,23,24,25,26,27,28,41,42,43,59,60,61,77,78,79]
        self.obs_indices["policy_HR"] = [0,1,2,3,4,20,21,22,23,24,25,26,27,28,44,45,46,62,63,64,80,81,82]
        super().__init__(config)
            
    @staticmethod
    def return_policies():
        obs_space = spaces.Box(-np.inf, np.inf, (23,), np.float64)
        policies = {
            SixLeggedFullyDecentralizedEnv.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
            SixLeggedFullyDecentralizedEnv.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
            SixLeggedFullyDecentralizedEnv.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
            SixLeggedFullyDecentralizedEnv.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
            SixLeggedFullyDecentralizedEnv.policy_names[4]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
            SixLeggedFullyDecentralizedEnv.policy_names[5]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
        }
        return policies
        
class SixLegged_Dec_AllInf_Env(SixLeggedControllerSuperEnv):
    """
    """    
    
    # This is ordering of the policies as applied here:
    policy_names = ["policy_FL","policy_FR",
        "policy_ML","policy_MR",
        "policy_HL","policy_HR",]
    
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
        self.obs_indices["policy_FL"] = range(0,83) #[0,1,2,3,4, 5, 6, 7,23,24,25,26,27,28,29,30,31,47,48,49,65,66,67]
        self.obs_indices["policy_FR"] = range(0,83) #[0,1,2,3,4, 8, 9,10,23,24,25,26,27,28,32,33,34,50,51,52,68,69,70]
        self.obs_indices["policy_ML"] = range(0,83) #[0,1,2,3,4,11,12,13,23,24,25,26,27,28,35,36,37,53,54,55,71,72,73]
        self.obs_indices["policy_MR"] = range(0,83) #[0,1,2,3,4,14,15,16,23,24,25,26,27,28,38,39,40,56,57,58,74,75,76]
        self.obs_indices["policy_HL"] = range(0,83) #[0,1,2,3,4,17,18,19,23,24,25,26,27,28,41,42,43,59,60,61,77,78,79]
        self.obs_indices["policy_HR"] = range(0,83) #[0,1,2,3,4,20,21,22,23,24,25,26,27,28,44,45,46,62,63,64,80,81,82]
        super().__init__(config)
            
    @staticmethod
    def return_policies():
        obs_space = spaces.Box(-np.inf, np.inf, (83,), np.float64)
        policies = {
            SixLeggedFullyDecentralizedEnv.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
            SixLeggedFullyDecentralizedEnv.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
            SixLeggedFullyDecentralizedEnv.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
            SixLeggedFullyDecentralizedEnv.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
            SixLeggedFullyDecentralizedEnv.policy_names[4]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
            SixLeggedFullyDecentralizedEnv.policy_names[5]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.]), np.array([+1.,+1.,+1.])), {}),
        }
        return policies