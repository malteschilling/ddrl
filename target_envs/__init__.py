from gym.envs.registration import registry, register, make, spec
from ray.tune.registry import register_env
from gym.wrappers.time_limit import TimeLimit
from target_envs.quantruped_v3 import QuAntrupedEnv

from target_envs.quantruped_adaptor_multi_environment import QuantrupedMultiPoliciesEnv
from target_envs.quantruped_fourDecentralizedController_environments import Quantruped_Local_Env

register(
	id='QuAntruped-v3',
	entry_point='target_envs.quantruped_v3:QuAntrupedEnv',
	max_episode_steps=1000,
	reward_threshold=6000.0,
)

register_env("QuAntruped-v3", lambda config: TimeLimit(QuAntrupedEnv(), max_episode_steps=1000))

register_env("QuantrupedMultiEnv_Centralized", lambda config: QuantrupedMultiPoliciesEnv(config) )
register_env("QuantrupedMultiEnv_Local", lambda config: Quantruped_Local_Env(config) )