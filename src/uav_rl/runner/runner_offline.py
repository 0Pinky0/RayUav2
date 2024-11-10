import glob
from pathlib import Path

import ray
import yaml
from ray import tune, air
from ray.rllib.core.rl_module import RLModuleSpec

from uav_rl.models.uav_encoder import get_catalog
import uav_rl.runner.common  # noqa
from uav_rl.runner.common import get_config_cls

config_id = 'uav-dqn-offline.yaml'
config_dict = yaml.load(open(f'../conf/{config_id}'), Loader=yaml.FullLoader)
if __name__ == '__main__':
    algo_config = (
        get_config_cls(config_dict['algo_name'])()
        .framework(framework='torch')
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .resources(
            num_cpus_for_main_process=16,
        )
        .rollouts(num_env_runners=0)
        .learners(
            num_learners=1,
            num_gpus_per_learner=1,
        )
        .evaluation(
            evaluation_num_workers=1,
            evaluation_interval=10,
            evaluation_duration="auto",
            evaluation_duration_unit="episodes",
            evaluation_parallel_to_training=True,
        )
    )
    if 'environment' in config_dict:
        algo_config = algo_config.environment(**config_dict['environment'])
    if 'training' in config_dict:
        algo_config = algo_config.training(**config_dict['training'])
    rl_module_dict = {
        'catalog_class': get_catalog(config_dict['algo_name']),
        # 'catalog_class': UavDQNCatalog,
        'load_state_path': config_dict['ckpt'],
    }
    algo_config = algo_config.rl_module(rl_module_spec=RLModuleSpec(**rl_module_dict))
    algo_config = algo_config.offline_data(
        input_="dataset",
        input_config={
            'format': 'json',
            'paths': glob.glob(f'{Path(__file__).parent.parent}/data/astar-out/*.json'),
        },
    )
    tuner = tune.Tuner(
        config_dict['algo_name'],
        param_space=algo_config,
        run_config=air.RunConfig(
            stop=config_dict['stop'],
            verbose=2,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=5,
                checkpoint_at_end=True,
            ),
        ),
    )
    results = tuner.fit()
    print(results)
    ray.shutdown()
