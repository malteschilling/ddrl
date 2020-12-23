import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces

from simulation_envs import QuantrupedMultiPoliciesEnv

class QuantrupedFullyDecentralizedGlobalCostEnv(QuantrupedMultiPoliciesEnv):
    """
    """    
    
    # This is ordering of the policies as applied here:
    policy_names = ["policy_FL","policy_HL","policy_HR","policy_FR"]
    
    def __init__(self, config):
        self.obs_indices = {}
        # First global information: 
        # 0: height, 1-4: quaternion orientation torso
        # 5: hip FL angle, 6: knee FL angle
        # 7: hip HL angle, 8: knee HL angle
        # 9: hip HR angle, 10: knee HR angle
        # 11: hip FR angle, 12: knee FR angle
        # Velocities follow same ordering, but have in addition x and y vel.
        # 13-15: vel, 16-18: rotational velocity
        # 19: hip FL angle, 20: knee FL angle
        # 21: hip HL angle, 22: knee HL angle
        # 23: hip HR angle, 24: knee HR angle
        # 25: hip FR angle, 26: knee FR angle
        # Passive forces same ordering, only local information used
        # 27: hip FL angle, 28: knee FL angle
        # 29: hip HL angle, 30: knee HL angle
        # 31: hip HR angle, 32: knee HR angle
        # 33: hip FR angle, 34: knee FR angle
        # Last: control signals (clipped) from last time step
        # Unfortunately, different ordering (as the action spaces...)
        # 37: hip FL angle, 38: knee FL angle
        # 39: hip HL angle, 40: knee HL angle
        # 41: hip HR angle, 42: knee HR angle
        # 35: hip FR angle, 36: knee FR angle
        self.obs_indices["policy_FL"] = [0,1,2,3,4, 5, 6,13,14,15,16,17,18,19,20,27,28,37,38]
        self.obs_indices["policy_HL"] = [0,1,2,3,4, 7, 8,13,14,15,16,17,18,21,22,29,30,39,40]
        self.obs_indices["policy_HR"] = [0,1,2,3,4, 9,10,13,14,15,16,17,18,23,24,31,32,41,42]
        self.obs_indices["policy_FR"] = [0,1,2,3,4,11,12,13,14,15,16,17,18,25,26,33,34,35,36]
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

#     def distribute_contact_cost(self):
#         contact_cost = {}
#         raw_contact_forces = self.env.sim.data.cfrc_ext
#         contact_forces = np.clip(raw_contact_forces, -1., 1.) 
#         contact_cost[self.policy_names[0]] = 0.25 * self.env.contact_cost_weight * np.sum(
#             np.square(contact_forces)) #global_contact_costs + np.sum(contact_costs[2:5])
#         contact_cost[self.policy_names[1]] = 0.25 * self.env.contact_cost_weight * np.sum(
#             np.square(contact_forces)) #global_contact_costs + np.sum(contact_costs[5:8])
#         contact_cost[self.policy_names[2]] = 0.25 * self.env.contact_cost_weight * np.sum(
#             np.square(contact_forces)) #global_contact_costs + np.sum(contact_costs[8:11])
#         contact_cost[self.policy_names[3]] = 0.25 * self.env.contact_cost_weight * np.sum(
#             np.square(contact_forces)) #global_contact_costs + np.sum(contact_costs[11:])
#         #print(contact_cost)
#         #sum_c = 0.
#         #for i in self.policy_names:
#          #   sum_c += contact_cost[i]
#         #print("Calculated: ", sum_c)
#         return contact_cost
        
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
        contact_cost[self.policy_names[3]] = global_contact_costs + np.sum(contact_costs[11:])
        #print(contact_cost)
        #sum_c = 0.
        #for i in self.policy_names:
         #   sum_c += contact_cost[i]
        #print("Calculated: ", sum_c)
        return contact_cost

    def distribute_reward(self, reward_full, info, action_dict):
        fw_reward = info['reward_forward']
        rew = {}
        contact_costs = self.distribute_contact_cost()

        # Compute control costs:
        sum_control_cost = 0
        for policy_name in self.policy_names:
            sum_control_cost += np.sum(np.square(action_dict[policy_name]))
        #print("COSTs: ", sum_control_cost, " / ", action_dict)
        for policy_name in self.policy_names:
            rew[policy_name] = (fw_reward / len(self.policy_names)) \
                - (self.env.ctrl_cost_weight * 0.25 * sum_control_cost) \
                - (contact_costs[policy_name])
        return rew

    def concatenate_actions(self, action_dict):
        # Return actions in the (DIFFERENT in Mujoco) order FR - FL - HL - HR
        actions = np.concatenate( (action_dict[self.policy_names[3]],
            action_dict[self.policy_names[0]],
            action_dict[self.policy_names[1]],
            action_dict[self.policy_names[2]]) )
        return actions

    @staticmethod
    def return_policies(obs_space):
        obs_space = spaces.Box(-np.inf, np.inf, (19,), np.float64)
        policies = {
            QuantrupedFullyDecentralizedGlobalCostEnv.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralizedGlobalCostEnv.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralizedGlobalCostEnv.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralizedGlobalCostEnv.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
        }
        return policies
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        if agent_id.startswith("policy_FL"):
            return "policy_FL"
        elif agent_id.startswith("policy_HL"):
            return "policy_HL"
        elif agent_id.startswith("policy_HR"):
            return "policy_HR"
        else:
            return "policy_FR" 
