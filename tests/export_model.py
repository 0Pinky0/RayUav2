import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule
from ray.tune.registry import register_env

from uav_envs.uav_env_v7 import UavEnvironment
from uav_envs.wrappers.raster_wrapper import RasterWrapper


ckpt_name = 'DQN_2024-10-22_16-38-44/DQN_UavEnv_0e7f3_00000_0_2024-10-22_16-38-48/checkpoint_000002'
# ckpt_name = 'DQN_2024-10-15_17-50-48/DQN_UavEnv_f69bf_00000_0_2024-10-15_17-50-51/checkpoint_000040'
# ckpt_name = 'DQN_2024-10-16_14-09-05/DQN_UavEnv_27d50_00000_0_2024-10-16_14-09-08/checkpoint_000071'
ckpt = f'/home/wjl/ray_results/{ckpt_name}/learner_group/learner/rl_module/default_policy'

register_env("UavEnv", lambda cfg: RasterWrapper(UavEnvironment(**cfg)))
rl_module = RLModule.from_checkpoint(ckpt)
rl_module.input_specs_train()
env_config = {
    "dimensions": (1000, 1000),
    "fixed_obstacles": 10,
    "dynamic_obstacles": 10,
    "occur_obstacles": 1,
    "occur_number_max": 3,
    "return_raster": True,
    "prevent_stiff": False,
    "use_lidar": True,
    "draw_lidar": False,
    "lidar_range": 250.0,
    "lidar_rays": 42,
    "field_of_view": 210.0,
    "center_obstacles": True
}
env = RasterWrapper(UavEnvironment(**env_config))
sample_batch = {
    "obs": torch.from_numpy(env.observation_space.sample()).float().unsqueeze(0),
    "actions": torch.from_numpy(np.array(env.action_space.sample())).float().unsqueeze(0),
    "new_obs": torch.from_numpy(env.observation_space.sample()).float().unsqueeze(0),
}

# Must be called before forward
rl_module.forward = rl_module.forward_inference
# ONNX
# torch.save(rl_module, "evolve_model.pth")
torch.onnx.export(rl_module,
                  {"batch": sample_batch},
                  "evolve_model.onnx",
                  verbose=True,
                  opset_version=16)
