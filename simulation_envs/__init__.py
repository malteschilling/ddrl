from gym.envs.registration import registry, register, make, spec
from ray.tune.registry import register_env
from gym.wrappers.time_limit import TimeLimit
from simulation_envs.quantruped_v3 import QuAntrupedEnv
from simulation_envs.ant_v3_mujoco_2 import AntEnvMujoco2

from simulation_envs.quantruped_adaptor_multi_environment import QuantrupedMultiPoliciesEnv
from simulation_envs.quantruped_fourDecentralizedController_environments import QuantrupedFullyDecentralizedEnv
from simulation_envs.quantruped_fourDecentralizedController_environments import Quantruped_LocalSingleNeighboringLeg_Env
from simulation_envs.quantruped_fourDecentralizedController_environments import Quantruped_LocalSingleDiagonalLeg_Env
from simulation_envs.quantruped_fourDecentralizedController_environments import Quantruped_Local_Env
from simulation_envs.quantruped_twoDecentralizedController_environments import Quantruped_TwoSideControllers_Env
from simulation_envs.quantruped_twoDecentralizedController_environments import Quantruped_TwoDiagControllers_Env

register(
	id='QuAntruped-v3',
	entry_point='simulation_envs.quantruped_v3:QuAntrupedEnv',
	max_episode_steps=1000,
	reward_threshold=6000.0,
)

register_env("QuAntruped-v3", lambda config: TimeLimit(QuAntrupedEnv(), max_episode_steps=1000))

register(
	id='Ant_Muj2-v3',
	entry_point='simulation_envs.ant_v3_mujoco_2:AntEnvMujoco2',
	max_episode_steps=1000,
	reward_threshold=6000.0,
)

register_env("Ant_Muj2-v3", lambda config: TimeLimit(AntEnvMujoco2(), max_episode_steps=1000))

register_env("QuantrupedMultiEnv_Centralized", lambda config: QuantrupedMultiPoliciesEnv(config) )

register_env("QuantrupedMultiEnv_FullyDecentral", lambda config: QuantrupedFullyDecentralizedEnv(config) )

register_env("QuantrupedMultiEnv_SingleNeighbor", lambda config: Quantruped_LocalSingleNeighboringLeg_Env(config) )
register_env("QuantrupedMultiEnv_SingleDiagonal", lambda config: Quantruped_LocalSingleDiagonalLeg_Env(config) )
register_env("QuantrupedMultiEnv_Local", lambda config: Quantruped_Local_Env(config) )
register_env("QuantrupedMultiEnv_TwoSides", lambda config: Quantruped_TwoSideControllers_Env(config) )
register_env("QuantrupedMultiEnv_TwoDiags", lambda config: Quantruped_TwoDiagControllers_Env(config) )