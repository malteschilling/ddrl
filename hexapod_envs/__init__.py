from gym.envs.registration import registry, register, make, spec
from ray.tune.registry import register_env
from gym.wrappers.time_limit import TimeLimit
from hexapod_envs.hexapod import HexapodEnv

from hexapod_envs.hexapod_adaptor_multi_environment import HexapodMultiPoliciesEnv
from hexapod_envs.hexapod_centralizedController_environment import Hexapod_Centralized_Env
from hexapod_envs.hexapod_sixDecentralizedController_environments import HexapodFullyDecentralizedEnv
from hexapod_envs.hexapod_sixDecentralizedController_environments import Hexapod_Local_Env

from hexapod_envs.hexapod_twoDecentralizedController_environments import Hexapod_TwoSideControllers_Env

#from hexapod_envs.hexapod_deploy_default import Hexapod
from hexapod_envs.hexapod_trossen_adapt import Hexapod

from hexapod_envs.hexapod_trossen_adapt_ms import PhantomX
from hexapod_envs.phantomX_adaptor_multi_environment import PhantomXMultiPoliciesEnv
from hexapod_envs.phantomX_centralizedController_environment import PhantomX_Centralized_Env

register(
	id='Hexapod-v1',
	entry_point='hexapod_envs.hexapod:HexapodEnv',
	max_episode_steps=1000,
	reward_threshold=4000.0,
)
register_env("Hexapod-v1", lambda config: TimeLimit(HexapodEnv(), max_episode_steps=1000))

from hexapod_envs.ant_six import AntSixEnv

register(
	id='AntSix-v1',
	entry_point='hexapod_envs.ant_six:AntSixEnv',
	max_episode_steps=300,
	reward_threshold=900.0,
)
register_env("AntSix-v1", lambda config: TimeLimit(AntSixEnv(), max_episode_steps=300))


register(
	id='PhantomX-v1',
	entry_point='hexapod_envs.hexapod_trossen_adapt_ms:PhantomX',
	max_episode_steps=1000,
	reward_threshold=4000.0,
)

register(
	id='Nexapod-v1',
	entry_point='hexapod_envs.hexapod_trossen_adapt:Hexapod',
	max_episode_steps=1000,
	reward_threshold=4000.0,
)

#register_env("PhantomX-v1", lambda config: TimeLimit(PhantomX(), max_episode_steps=1000))
register_env("Nexapod-v1", lambda config: TimeLimit(Hexapod(), max_episode_steps=1000))

#register_env("PhantomXMultiEnv_Centralized", lambda config: PhantomXMultiPoliciesEnv(config) )
register_env("PhantomXMultiEnv_Centralized", lambda config: PhantomX_Centralized_Env(config) )

register_env("HexapodMultiEnv_FullyDecentral", lambda config: HexapodFullyDecentralizedEnv(config) )
register_env("HexapodMultiEnv_Local", lambda config: Hexapod_Local_Env(config) )
register_env("HexapodMultiEnv_TwoSides", lambda config: Hexapod_TwoSideControllers_Env(config) )
# register_env("QuantrupedMultiEnv_TwoDiags", lambda config: Quantruped_TwoDiagControllers_Env(config) )