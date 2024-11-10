import ray
import yaml
from ray import tune, air
from ray.rllib.core.rl_module import RLModuleSpec

import uav_rl.runner.common  # noqa
from uav_rl.models.uav_encoder import get_catalog
from uav_rl.runner.common import get_config_cls
from uav_rl.runner.common.flatten_obs_with_original_space import FlattenObservationsWithOriginalSpace

config_id = 'uav-vec-sac.yaml'
# config_id = 'cartpole-ppo.yaml'
config_dict = yaml.load(open(f'../conf/{config_id}'), Loader=yaml.FullLoader)
if __name__ == '__main__':
    algo_config = (
        get_config_cls(config_dict['algo_name'])()
        .framework(framework='torch')
        .env_runners(
            num_env_runners=8,
            num_cpus_per_env_runner=2,
            # env_to_module_connector=lambda env:
            # FlattenObservationsWithOriginalSpace(multi_agent=False)
        )
        .api_stack(
            # enable_rl_module_and_learner=True,
            # enable_env_runner_and_connector_v2=True,
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .resources(
            num_cpus_for_main_process=4,
        )
        .learners(
            num_learners=1,
            num_gpus_per_learner=1,
        )
        .evaluation(
            evaluation_num_env_runners=1,
            evaluation_interval=10,
            # evaluation_duration="auto",
            evaluation_duration=5,
            # evaluation_duration_unit="timesteps",
            evaluation_duration_unit="episodes",
            evaluation_parallel_to_training=False,
        )
    )
    if 'environment' in config_dict:
        algo_config = algo_config.environment(**config_dict['environment'])
    if 'training' in config_dict:
        algo_config = algo_config.training(**config_dict['training'])
    if 'rl_module' in config_dict:
        algo_config = algo_config.rl_module(**config_dict['rl_module'])
    rl_module_dict = {
        'catalog_class': get_catalog(config_dict['algo_name']),
        'load_state_path': config_dict['ckpt'],
    }
    # algo_config = algo_config.rl_module(
    #     model_config_dict=dict(
    #         {
    #             # "vf_share_layers": True,
    #             "conv_filters": [[16, 4, 2], [32, 4, 2], [64, 4, 2], [128, 4, 2]],
    #             "conv_activation": "relu",
    #             "post_fcnet_hiddens": [256],
    #             "uses_new_env_runners": True,
    #         },
    #     )
    # )
    # algo_config = algo_config.rl_module(rl_module_spec=RLModuleSpec(**rl_module_dict))
    # results = algo_config.build().train()
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
