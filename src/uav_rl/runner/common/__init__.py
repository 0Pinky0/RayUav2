from ray.rllib.env.wrappers.atari_wrappers import wrap_atari_for_new_api_stack
from ray.tune import register_env

from uav_envs import UavEnvironment
from uav_envs.wrappers.raster_wrapper import RasterWrapper
from .pixel_cartpole import pixel_cartpole
from .utils import get_config_cls
import gymnasium as gym

register_env(
    'UavEnv',
    lambda cfg: RasterWrapper(
        UavEnvironment(**cfg)
    )
)
register_env(
    'UavEnvVec',
    lambda cfg: UavEnvironment(**cfg)
)
register_env(
    'Ray/Pong-v5',
    lambda cfg: wrap_atari_for_new_api_stack(
        gym.make('ALE/Pong-v5', **cfg),
        frameskip=4,
        framestack=4,
    )
)
register_env(
    'Ray/SpaceInvaders-v5',
    lambda cfg: wrap_atari_for_new_api_stack(
        gym.make('ALE/SpaceInvaders-v5', **cfg),
        frameskip=4,
        framestack=4,
    )
)
register_env(
    'CartPolePixels',
    pixel_cartpole,
)
