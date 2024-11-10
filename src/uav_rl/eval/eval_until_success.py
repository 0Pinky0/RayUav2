import time

import numpy as np
import torch
from gymnasium.wrappers import HumanRendering
from ray.rllib.core.rl_module import RLModule

from uav_envs.wrappers.raster_wrapper import RasterWrapper
import uav_envs  # noqa
import gymnasium as gym

# render = True
render = False

ckpt_name = 'DQN_2024-10-15_17-50-48/DQN_UavEnv_f69bf_00000_0_2024-10-15_17-50-51/checkpoint_000068'
# ckpt_name = 'DQN_2024-10-16_14-09-05/DQN_UavEnv_27d50_00000_0_2024-10-16_14-09-08/checkpoint_000071'
rlmodule_ckpt = f'/home/wjl/ray_results/{ckpt_name}/learner_group/learner/rl_module/default_policy'
loaded_module = RLModule.from_checkpoint(
    rlmodule_ckpt
)

env = RasterWrapper(
    gym.make(
        "UavEnv-v7",
        render_mode='rgb_array',
        # center_obstacles=True,
    )
)
if render:
    env = HumanRendering(env)

total_ep = 20
success_ep = 0
info = {'done': ''}

costs = [0.]
i = 0
seed = 0
while i < total_ep:
    done = False
    obs, _ = env.reset(seed=seed)
    if render:
        env.render()
    while not done:
        obs = {'obs': torch.from_numpy(obs).unsqueeze(0)}
        tic = time.time()
        action = loaded_module.forward_inference(obs)['actions'].item()
        costs[i] += time.time() - tic
        obs, reward, terminated, truncated, info = env.step(action)
        if render:
            env.render()
        done = truncated or terminated
    if info['done'] == 'goal_reached':
        success_ep += 1
        i += 1
        costs.append(0.)
    else:
        costs[i] = 0.
    seed += 1
    print(
        f'ep {i} / {total_ep}: success = {success_ep} / {total_ep} = {success_ep / total_ep}\n\tcosts = {np.mean(costs):.5f}')
