from gym.envs.registration import registry, register, make, spec
from ray.tune.registry import register_env
from gym.wrappers.time_limit import TimeLimit
from hexapod_envs.Hexapod_v1 import HexapodEnv

from hexapod_envs.hexapod_centralizedController_environment import HexapodMultiEnv_Centralized_Env
from hexapod_envs.hexapod_decentralizedController_environments import HexapodFullyDecentralizedEnv
from hexapod_envs.hexapod_decentralizedController_environments import Hexapod_Local_Env

register(
	id='Hexapod-v1',
	entry_point='hexapod_envs.Hexapod_v1:HexapodEnv',
	max_episode_steps=1000,
	reward_threshold=1000.0,
)
register_env("Hexapod-v1", lambda config: TimeLimit(HexapodEnv(), max_episode_steps=1000))

register_env("HexapodMultiEnv_Centralized", lambda config: HexapodMultiEnv_Centralized_Env(config))
register_env("HexapodMultiEnv_FullyDecentral", lambda config: HexapodFullyDecentralizedEnv(config))
register_env("HexapodMultiEnv_Local", lambda config: Hexapod_Local_Env(config))