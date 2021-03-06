import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces

from target_envs.quantruped_adaptor_multi_environment import QuantrupedMultiPolicies_TVel_Env

class QuantrupedFourControllerSuper_TVel_Env(QuantrupedMultiPolicies_TVel_Env):
    """ A decentralized controller for the quantruped agent.
    
        This environment is splitting a quantruped agent into 
        four individual controllers, one for each leg
        and distributes all available information. 
        
        Reward: is aiming for a given target velocity.
        
        This is the general parent class - the derived classes deal with how   
        to distribute information to each policy.
        
        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in derived classes and differs between the different architectures.
        - distribute_contact_cost: how to distribute (contact) costs individually to controllers 
        - concatenate_actions: how to integrate the control signals from the controllers

    """  

    # Distribute the observations into the decentralized policies.
    # In the trivial case, only a single policy is used that gets all information.
    def distribute_observations(self, obs_full):
        """ 
        Construct dictionary that routes to each policy only the relevant
        local information.
        """
        obs_distributed = {}
        for policy_name in self.policy_names:
            obs_distributed[policy_name] = obs_full[self.obs_indices[policy_name],]
        return obs_distributed

    # Distribute the contact costs into the decentralized policies.
    # In the trivial case, only a single policy is used that gets all information. 
    def distribute_contact_cost(self):
        contact_cost = {}
        #from mujoco_py import functions
        #functions.mj_rnePostConstraint(self.env.model, self.env.data)
        raw_contact_forces = self.env.sim.data.cfrc_ext
        contact_forces = np.clip(raw_contact_forces, -1., 1.)
        contact_costs = self.env.contact_cost_weight * np.square(contact_forces)
        global_contact_costs = np.sum(contact_costs[0:2])/4.
        contact_cost[self.policy_names[0]] = global_contact_costs + np.sum(contact_costs[2:5])
        contact_cost[self.policy_names[1]] = global_contact_costs + np.sum(contact_costs[5:8])
        contact_cost[self.policy_names[2]] = global_contact_costs + np.sum(contact_costs[8:11])
        contact_cost[self.policy_names[3]] = global_contact_costs + np.sum(contact_costs[11:])
        return contact_cost
        
    def concatenate_actions(self, action_dict):
        # Return actions in the (DIFFERENT in Mujoco) order FR - FL - HL - HR
        actions = np.concatenate( (action_dict[self.policy_names[3]],
            action_dict[self.policy_names[0]],
            action_dict[self.policy_names[1]],
            action_dict[self.policy_names[2]]) )
        return actions
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        if agent_id.startswith("policy_FL"):
            return "policy_FL"
        elif agent_id.startswith("policy_HL"):
            return "policy_HL"
        elif agent_id.startswith("policy_HR"):
            return "policy_HR"
        else:
            return "policy_FR" 


class QuantrupedFullyDecentralized_TVel_Env(QuantrupedFourControllerSuper_TVel_Env):
    """ A decentralized controller for the quantruped agent.
    
        Reward: is aiming for a given target velocity.
    
        For the fully decentralized case, only information from that particular leg
        is used as input to the decentralized policies.
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
        # 43: target_velocity
        self.obs_indices["policy_FL"] = [0,1,2,3,4, 5, 6,13,14,15,16,17,18,19,20,27,28,37,38,43]
        self.obs_indices["policy_HL"] = [0,1,2,3,4, 7, 8,13,14,15,16,17,18,21,22,29,30,39,40,43]
        self.obs_indices["policy_HR"] = [0,1,2,3,4, 9,10,13,14,15,16,17,18,23,24,31,32,41,42,43]
        self.obs_indices["policy_FR"] = [0,1,2,3,4,11,12,13,14,15,16,17,18,25,26,33,34,35,36,43]
        super().__init__(config)
            
    @staticmethod
    def return_policies():
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (20,), np.float64)
        policies = {
            QuantrupedFullyDecentralized_TVel_Env.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralized_TVel_Env.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralized_TVel_Env.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralized_TVel_Env.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
        }
        return policies
        
class Quantruped_Local_TVel_Env(QuantrupedFourControllerSuper_TVel_Env):
    """ A decentralized controller for the quantruped agent.
    
        Reward: is aiming for a given target velocity.
    
        For the local, decentralized case, information from that particular leg
        and the two neighboring legs is used as input to the decentralized policies.
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
        # FL also gets local information from HL and FR
        # 43: target_velocity
        self.obs_indices["policy_FL"] = [0,1,2,3,4, 5, 6, 7, 8,11,12,13,14,15,16,17,18,19,20,21,22,25,26,27,28,29,30,33,34,37,38,39,40,35,36,43]
        # HL also gets local information from HR and FL
        self.obs_indices["policy_HL"] = [0,1,2,3,4, 7, 8, 9,10, 5, 6,13,14,15,16,17,18,21,22,23,24,19,20,29,30,31,32,27,28,39,40,41,42,37,38,43]
        # HR also gets local information from FR and HL
        self.obs_indices["policy_HR"] = [0,1,2,3,4, 9,10,11,12, 7, 8,13,14,15,16,17,18,23,24,25,26,21,22,31,32,33,34,29,30,41,42,35,36,39,40,43]
        # FR also gets local information from FL and HR
        self.obs_indices["policy_FR"] = [0,1,2,3,4,11,12, 5, 6, 9,10,13,14,15,16,17,18,25,26,19,20,23,24,33,34,27,28,31,32,35,36,37,38,41,42,43]
        super().__init__(config)
        
            
    @staticmethod
    def return_policies():
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (36,), np.float64)
        policies = {
            Quantruped_Local_TVel_Env.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_Local_TVel_Env.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_Local_TVel_Env.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_Local_TVel_Env.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
        }
        return policies
