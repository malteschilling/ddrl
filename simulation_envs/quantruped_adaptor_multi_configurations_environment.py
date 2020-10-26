import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces

from simulation_envs import QuantrupedMultiPoliciesEnv

class QuantrupedFullyDecentralizedEnv(QuantrupedMultiPoliciesEnv):
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
        
    def concatenate_actions(self, action_dict):
        # Return actions in the (DIFFERENT in Mujoco) order FR - FL - HL - HR
        actions = np.concatenate( (action_dict[self.policy_names[3]],
            action_dict[self.policy_names[0]],
            action_dict[self.policy_names[1]],
            action_dict[self.policy_names[2]]) )
        return actions
        
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
            
    @staticmethod
    def return_policies(obs_space):
        obs_space = spaces.Box(-np.inf, np.inf, (19,), np.float64)
        policies = {
            QuantrupedFullyDecentralizedEnv.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralizedEnv.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralizedEnv.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralizedEnv.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
        }
        return policies
        
class Quantruped_LocalSingleNeighboringLeg_Env(QuantrupedMultiPoliciesEnv):
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
        # FL also gets local information from HL
        self.obs_indices["policy_FL"] = [0,1,2,3,4, 5, 6, 7, 8,13,14,15,16,17,18,19,20,21,22,27,28,29,30,37,38,39,40]
        # HL also gets local information from HR
        self.obs_indices["policy_HL"] = [0,1,2,3,4, 7, 8, 9,10,13,14,15,16,17,18,21,22,23,24,29,30,31,32,39,40,41,42]
        # HR also gets local information from FR
        self.obs_indices["policy_HR"] = [0,1,2,3,4, 9,10,11,12,13,14,15,16,17,18,23,24,25,26,31,32,33,34,41,42,35,36]
        # FR also gets local information from FL
        self.obs_indices["policy_FR"] = [0,1,2,3,4,11,12, 5, 6,13,14,15,16,17,18,25,26,19,20,33,34,27,28,35,36,37,38]
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
        
    def concatenate_actions(self, action_dict):
        # Return actions in the (DIFFERENT in Mujoco) order FR - FL - HL - HR
        actions = np.concatenate( (action_dict[self.policy_names[3]],
            action_dict[self.policy_names[0]],
            action_dict[self.policy_names[1]],
            action_dict[self.policy_names[2]]) )
        return actions
        
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
            
    @staticmethod
    def return_policies(obs_space):
        obs_space = spaces.Box(-np.inf, np.inf, (27,), np.float64)
        policies = {
            Quantruped_LocalSingleNeighboringLeg_Env.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_LocalSingleNeighboringLeg_Env.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_LocalSingleNeighboringLeg_Env.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_LocalSingleNeighboringLeg_Env.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
        }
        return policies

class Quantruped_LocalSingleDiagonalLeg_Env(QuantrupedMultiPoliciesEnv):
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
        self.obs_indices["policy_FL"] = [0,1,2,3,4, 5, 6, 9,10,13,14,15,16,17,18,19,20,23,24,27,28,31,32,37,38,41,42]
        self.obs_indices["policy_HL"] = [0,1,2,3,4, 7, 8,11,12,13,14,15,16,17,18,21,22,25,26,29,30,33,34,39,40,35,36]
        self.obs_indices["policy_HR"] = self.obs_indices["policy_FL"] #[0,1,2,3,4, 9,10,13,14,15,16,17,18,23,24,31,32,41,42]
        self.obs_indices["policy_FR"] = self.obs_indices["policy_HL"] #[0,1,2,3,4,11,12,13,14,15,16,17,18,25,26,33,34,35,36]
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
        
    def concatenate_actions(self, action_dict):
        # Return actions in the (DIFFERENT in Mujoco) order FR - FL - HL - HR
        actions = np.concatenate( (action_dict[self.policy_names[3]],
            action_dict[self.policy_names[0]],
            action_dict[self.policy_names[1]],
            action_dict[self.policy_names[2]]) )
        return actions
        
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
            
    @staticmethod
    def return_policies(obs_space):
        obs_space = spaces.Box(-np.inf, np.inf, (27,), np.float64)
        policies = {
            Quantruped_LocalSingleDiagonalLeg_Env.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_LocalSingleDiagonalLeg_Env.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_LocalSingleDiagonalLeg_Env.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_LocalSingleDiagonalLeg_Env.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
        }
        return policies
        
class Quantruped_Local_Env(QuantrupedMultiPoliciesEnv):
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
        # FL also gets local information from HL and FR
        self.obs_indices["policy_FL"] = [0,1,2,3,4, 5, 6, 7, 8,11,12,13,14,15,16,17,18,19,20,21,22,25,26,27,28,29,30,33,34,37,38,39,40,35,36]
        # HL also gets local information from HR and FL
        self.obs_indices["policy_HL"] = [0,1,2,3,4, 7, 8, 9,10, 5, 6,13,14,15,16,17,18,21,22,23,24,19,20,29,30,31,32,27,28,39,40,41,42,37,38]
        # HR also gets local information from FR and HL
        self.obs_indices["policy_HR"] = [0,1,2,3,4, 9,10,11,12, 7, 8,13,14,15,16,17,18,23,24,25,26,21,22,31,32,33,34,29,30,41,42,35,36,39,40]
        # FR also gets local information from FL and HR
        self.obs_indices["policy_FR"] = [0,1,2,3,4,11,12, 5, 6, 9,10,13,14,15,16,17,18,25,26,19,20,23,24,33,34,27,28,31,32,35,36,37,38,41,42]
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
        
#    def distribute_reward(self, reward_full, info, action_dict):
 #       fw_reward = info['reward_forward']
  #      rew = {}      
   #     for policy_name in self.policy_names:
    #        rew[policy_name] = fw_reward / len(self.policy_names) - self.env.ctrl_cost_weight * np.sum(np.square(action_dict[policy_name]))
     #   return rew
        
    def concatenate_actions(self, action_dict):
        # Return actions in the (DIFFERENT in Mujoco) order FR - FL - HL - HR
        actions = np.concatenate( (action_dict[self.policy_names[3]],
            action_dict[self.policy_names[0]],
            action_dict[self.policy_names[1]],
            action_dict[self.policy_names[2]]) )
        return actions
        
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
            
    @staticmethod
    def return_policies(obs_space):
        obs_space = spaces.Box(-np.inf, np.inf, (35,), np.float64)
        policies = {
            Quantruped_Local_Env.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_Local_Env.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_Local_Env.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_Local_Env.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
        }
        return policies
        
class Quantruped_TwoSideControllers_Env(QuantrupedMultiPoliciesEnv):
    """
    """    
    
    # This is ordering of the policies as applied here:
    policy_names = ["policy_LEFT","policy_RIGHT"]
    
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
        # Each controller only gets information from that body side: Left
        self.obs_indices["policy_LEFT"] =  [0,1,2,3,4, 5, 6, 7, 8,13,14,15,16,17,18,19,20,21,22,27,28,29,30,37,38,39,40]
        # Each controller only gets information from that body side: Right
        self.obs_indices["policy_RIGHT"] = [0,1,2,3,4, 9,10,11,12,13,14,15,16,17,18,23,24,25,26,31,32,33,34,41,42,35,36]
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
        
    def concatenate_actions(self, action_dict):
        # Return actions in the (DIFFERENT in Mujoco) order FR - FL - HL - HR
        actions = np.concatenate( (action_dict[self.policy_names[1]][2:],
            action_dict[self.policy_names[0]][0:2],
            action_dict[self.policy_names[0]][2:],
            action_dict[self.policy_names[1]][0:2]) )
        return actions
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        if agent_id.startswith("policy_LEFT"):
            return "policy_LEFT"
        else:
            return "policy_RIGHT" 
            
    @staticmethod
    def return_policies(obs_space):
        obs_space = spaces.Box(-np.inf, np.inf, (27,), np.float64)
        policies = {
            Quantruped_TwoSideControllers_Env.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.,-1.]), np.array([+1.,+1.,+1.,+1.])), {}),
            Quantruped_TwoSideControllers_Env.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.,-1.]), np.array([+1.,+1.,+1.,+1.])), {})
        }
        return policies
        
class Quantruped_TwoDiagControllers_Env(QuantrupedMultiPoliciesEnv):
    """
    """    
    
    # This is ordering of the policies as applied here:
    policy_names = ["policy_FLHR","policy_HLFR"]
    
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
        # Each controller only gets information from two legs, diagonally arranged: FL-HR
        self.obs_indices["policy_FLHR"] = [0,1,2,3,4, 5, 6, 9,10,13,14,15,16,17,18,19,20,23,24,27,28,31,32,37,38,41,42]
        # Each controller only gets information from two legs, diagonally arranged: HL-FR
        self.obs_indices["policy_HLFR"] = [0,1,2,3,4, 7, 8,11,12,13,14,15,16,17,18,21,22,25,26,29,30,33,34,39,40,35,36]
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
        
    def concatenate_actions(self, action_dict):
        # Return actions in the (DIFFERENT in Mujoco) order FR - FL - HL - HR
        actions = np.concatenate( (action_dict[self.policy_names[1]][2:],
            action_dict[self.policy_names[0]][0:2],
            action_dict[self.policy_names[0]][2:],
            action_dict[self.policy_names[1]][0:2]) )
        return actions
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        if agent_id.startswith("policy_FLHR"):
            return "policy_FLHR"
        else:
            return "policy_HLFR" 
            
    @staticmethod
    def return_policies(obs_space):
        obs_space = spaces.Box(-np.inf, np.inf, (27,), np.float64)
        policies = {
            Quantruped_TwoDiagControllers_Env.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.,-1.]), np.array([+1.,+1.,+1.,+1.])), {}),
            Quantruped_TwoDiagControllers_Env.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.,-1.]), np.array([+1.,+1.,+1.,+1.])), {})
        }
        return policies