from gym.envs.registration import registry, register, make, spec
from ray.tune.registry import register_env
from gym.wrappers.time_limit import TimeLimit
from six_envs.sixlegged_v1 import SixLeggedEnv

from six_envs.sixlegged_adaptor_multi_environment import SixLeggedMultiPoliciesEnv

from six_envs.sixlegged_sixDecentralizedController_environments import SixLeggedFullyDecentralizedEnv
from six_envs.sixlegged_sixDecentralizedController_environments import SixLegged_Dec_AllInf_Env
from six_envs.sixlegged_centralizedController_environment import SixLegged_Centralized_Env

#from six_envs.quantruped_fourDecentralizedController_environments import Quantruped_LocalSingleNeighboringLeg_Env

#from six_envs.quantruped_fourDecentralizedController_environments import Quantruped_LocalSingleDiagonalLeg_Env

#from six_envs.quantruped_fourDecentralizedController_environments import Quantruped_Local_Env

#from six_envs.quantruped_fourDecentralizedController_environments import Quantruped_LocalSingleToFront_Env
#from six_envs.quantruped_twoDecentralizedController_environments import Quantruped_TwoSideControllers_Env
#from six_envs.quantruped_twoDecentralizedController_environments import Quantruped_TwoDiagControllers_Env
#from six_envs.quantruped_fourDecentralizedController_GlobalCosts_environments import QuantrupedFullyDecentralizedGlobalCostEnv

register(
	id='SixLegged-v1',
	entry_point='six_envs.sixlegged_v1:SixLeggedEnv',
	max_episode_steps=1000,
	reward_threshold=1000.0,
)

#register_env("QuAntruped-v3", lambda config: TimeLimit(QuAntrupedEnv(), max_episode_steps=1000))

register_env("SixLeggedMultiEnv_Centralized", lambda config: SixLegged_Centralized_Env(config) )

register_env("SixLeggedMultiEnv_FullyDecentral", lambda config: SixLeggedFullyDecentralizedEnv(config) )
register_env("SixLeggedMultiEnv_DecentralAllInf", lambda config: SixLegged_Dec_AllInf_Env(config) )

#register_env("QuantrupedMultiEnv_Local", lambda config: Quantruped_Local_Env(config) )