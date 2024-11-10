import gymnasium as gym
import numpy as np
from gymnasium.wrappers import PixelObservationWrapper, FrameStack, ResizeObservation, GrayScaleObservation


class NormalizedImageEnv(gym.ObservationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(
            -1.0,
            1.0,
            shape=self.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        return (observation.astype(np.float32) / 128.0) - 1.0


class PixelOnlyWrapper(gym.ObservationWrapper):
    def __init__(
            self,
            env: gym.Env,
    ):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = env.observation_space.spaces['pixels']

    def observation(self, observation):
        return observation['pixels']

def pixel_cartpole(cfg):
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = PixelObservationWrapper(env)
    env = PixelOnlyWrapper(env)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 64)
    env = NormalizedImageEnv(env)
    env = FrameStack(env, num_stack=4)
    return env

if __name__ == '__main__':
    env = pixel_cartpole({})
    obs, _ = env.reset()
