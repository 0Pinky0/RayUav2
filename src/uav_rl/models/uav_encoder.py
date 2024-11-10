from typing import Sequence, Type

import torch
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
import gymnasium as gym
from ray.rllib.algorithms.dqn.dqn_rainbow_catalog import DQNRainbowCatalog
from ray.rllib.core.models.base import Encoder, ENCODER_OUT
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.torch.base import TorchModel


# from uav_rl.models.templates import ConvNetBlock, SquashDims


class UavEncoderConfig(ModelConfig):
    output_dims = (256,)
    freeze = False

    def __init__(self, observation_space: gym.Space):
        self.observation_space = observation_space
        super().__init__()

    def build(self, framework):
        assert framework == "torch", "Unsupported framework `{}`!".format(framework)
        return UavEncoder(self)


class UavEncoder(TorchModel, Encoder):
    def __init__(self, config: UavEncoderConfig):
        super().__init__(config)
        self.obs_space = config.observation_space
        raster_shape: Sequence[int] = (16, 16, 16)
        cnn_channels: Sequence[int] = (16, 32, 64)
        kernel_sizes: Sequence[int] = (3, 3, 3)
        strides: Sequence[int] = (1, 1, 1)
        # vec_dim = 0
        vec_dim = 50
        vec_out = 256

        in_ch = raster_shape[0]
        layers = []
        for i in range(len(cnn_channels)):
            layers += [_ConvNetBlock(
                in_ch, cnn_channels[i], kernel_size=kernel_sizes[i], stride=strides[i]
            )]
            in_ch = cnn_channels[i]
        layers += [torch.nn.ReLU(inplace=True), SquashDims()]
        self.cnn_encoder = torch.nn.Sequential(*layers)
        cnn_output = self.cnn_encoder(torch.ones(raster_shape))
        self.post_encoder = nn.Sequential(
            nn.Linear(vec_dim + cnn_output.size(0), vec_out),
            nn.ReLU(),
            nn.Linear(vec_out, vec_out),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy_input = self.obs_space.sample()
            if isinstance(dummy_input, dict):
                for key, val in dummy_input.items():
                    dummy_input[key] = torch.from_numpy(val).unsqueeze(0)
            else:
                dummy_input = torch.from_numpy(dummy_input).unsqueeze(0)
            _ = self._forward(input_dict={'obs': dummy_input})

    def _forward(self, input_dict, **kwargs):
        obs_dict = restore_original_dimensions(
            input_dict['obs'],
            self.obs_space,
            'torch',
        )
        if not isinstance(obs_dict, dict):
            obs_dict = {'obs': obs_dict}
        embed = torch.concatenate([
            obs_dict['observation'],
            self.cnn_encoder(obs_dict['raster']),
        ], dim=-1)
        # embed = self.cnn_encoder(input_dict['obs'])
        embed = self.post_encoder(embed)
        return {ENCODER_OUT: embed}


algo_catalogs: dict[str, Type[Catalog]] = {
    'DQN': DQNRainbowCatalog,
    'PPO': PPOCatalog,
    'SAC': SACCatalog,
}


def get_catalog(name: str) -> Type[Catalog]:
    assert name in algo_catalogs
    AlgoCatalog = algo_catalogs[name]

    class UavEncoderCatalog(AlgoCatalog):
        @classmethod
        def _get_encoder_config(
                cls,
                observation_space: gym.Space,
                **kwargs,
        ):
            return UavEncoderConfig(observation_space)

    return UavEncoderCatalog
