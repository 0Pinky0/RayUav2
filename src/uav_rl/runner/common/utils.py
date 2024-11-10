from typing import Type

from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms import DQNConfig, PPOConfig, SACConfig

_config_class_mapping = {
    'DQN': DQNConfig,
    'PPO': PPOConfig,
    'SAC': SACConfig,
}


def get_config_cls(run_name: str) -> Type[AlgorithmConfig]:
    if run_name in _config_class_mapping:
        return _config_class_mapping[run_name]
    else:
        raise ValueError(f'Invalid run name: {run_name}')
