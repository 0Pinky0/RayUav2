import time

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import HumanRendering
from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule

import uav_envs  # noqa
from uav_envs.wrappers.raster_wrapper import RasterWrapper

render = True
# render = False

# ckpt_name = 'DQN_2024-10-18_20-08-02/DQN_UavEnv_a1340_00000_0_2024-10-18_20-08-04/checkpoint_000005'
# ckpt_name = 'DQN_2024-10-22_16-38-44/DQN_UavEnv_0e7f3_00000_0_2024-10-22_16-38-48/checkpoint_000003'
# ckpt_name = 'DQN_2024-10-15_17-50-48/DQN_UavEnv_f69bf_00000_0_2024-10-15_17-50-51/checkpoint_000040'
ckpt_name = 'PPO_2024-10-31_17-21-22\PPO_CartPolePixels_801be_00000_0_2024-10-31_17-21-24\checkpoint_000000'
# ckpt_name = 'DQN_2024-10-16_14-09-05/DQN_UavEnv_27d50_00000_0_2024-10-16_14-09-08/checkpoint_000071'
ckpt_name = ckpt_name.replace('\\', '/')
rlmodule_ckpt = f'/home/wjl/ray_results/{ckpt_name}/learner_group/learner/rl_module/default_policy'
loaded_module: TorchRLModule = RLModule.from_checkpoint(
    rlmodule_ckpt
)

# env = RasterWrapper(
#     gym.make(
#         "UavEnv-v7",
#         render_mode='rgb_array',
#         # center_obstacles=True,
#     )
# )
env = gym.make(
    "UavEnv-v7",
    render_mode='rgb_array',
    fixed_obstacles=20,
    dynamic_obstacles=20,
    occur_obstacles=1,
    occur_number_max=3,
    # draw_lidar=True,
    # center_obstacles=True,
)
# torch.onnx.export(
#     loaded_module,
#     ({'obs': env.observation_space.sample()},),
#     'model.onnx',
#     verbose=True,
# )
if render:
    env = HumanRendering(env)

total_ep = 20
success_ep = 0
info = {'done': ''}
costs = [0.]

for i in range(total_ep):
    print(
        f'ep {i} / {total_ep}: success = {success_ep} / {i + 1} = {success_ep / (i + 1)}\n\tcosts = {np.mean(costs):.5f}')
    if info['done'] == 'goal_reached':
        success_ep += 1
    done = False
    obs, _ = env.reset(seed=i)
    if render:
        env.render()
    while not done:
        # obs = {'obs': torch.from_numpy(obs).unsqueeze(0)}
        obs = {'obs': torch.from_numpy(obs['observation']).to(dtype=torch.float32).unsqueeze(0)}
        tic = time.time()
        action = loaded_module.forward_inference(obs)['actions'].item()
        costs[i] += time.time() - tic
        obs, reward, terminated, truncated, info = env.step(action)
        if render:
            env.render()
        done = truncated or terminated
    costs.append(0.)
if info['done'] == 'goal_reached':
    success_ep += 1
print(
    f'ep {total_ep} / {total_ep}: success = {success_ep} / {total_ep} = {success_ep / total_ep}\n\tcosts = {np.mean(costs):.5f}')
