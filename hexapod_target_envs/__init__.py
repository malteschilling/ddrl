from gym.envs.registration import registry, register, make, spec
from ray.tune.registry import register_env
from gym.wrappers.time_limit import TimeLimit

from hexapod_target_envs.ant_six import AntSixEnv
from hexapod_target_envs.ant_six_centralizedController_environment import AntSix_Centralized_Env

register(
	id='AntSix-v1',
	entry_point='hexapod_target_envs.ant_six:AntSixEnv',
	max_episode_steps=1000,
	reward_threshold=500.0,
)
register_env("AntSix-v1", lambda config: TimeLimit(AntSixEnv(), max_episode_steps=1000))

register_env("AntSix-Centralized", lambda config: AntSix_Centralized_Env(config))