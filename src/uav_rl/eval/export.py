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
ckpt_name = 'DQN_2024-10-22_16-38-44/DQN_UavEnv_0e7f3_00000_0_2024-10-22_16-38-48/checkpoint_000002'
# ckpt_name = 'DQN_2024-10-15_17-50-48/DQN_UavEnv_f69bf_00000_0_2024-10-15_17-50-51/checkpoint_000040'
# ckpt_name = 'DQN_2024-10-16_14-09-05/DQN_UavEnv_27d50_00000_0_2024-10-16_14-09-08/checkpoint_000071'
rlmodule_ckpt = f'/home/wjl/ray_results/{ckpt_name}/learner_group/learner/rl_module/default_policy'
loaded_module: TorchRLModule = RLModule.from_checkpoint(rlmodule_ckpt)

env = RasterWrapper(
    gym.make(
        "UavEnv-v7",
        render_mode='rgb_array',
        # center_obstacles=True,
    )
)
obs = env.observation_space.sample()
if isinstance(obs, dict):
    for k, v in obs.items():
        obs[k] = torch.from_numpy(v)
else:
    obs = torch.from_numpy(obs).unsqueeze(0)

torch.onnx.export(
    loaded_module,
    ({'batch': {
        'obs': obs
    }},),
    'model.onnx',
    verbose=True,
)
