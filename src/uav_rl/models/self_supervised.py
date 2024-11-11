from typing import Dict, Any

import torch
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.core import Columns
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.core.rl_module.apis import SelfSupervisedLossAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils import override
from ray.rllib.utils.typing import ModuleID
from torch import nn


class StateReconstructionLoss(TorchRLModule, SelfSupervisedLossAPI):
    @override(TorchRLModule)
    def setup(self):
        # Get the ICM achitecture settings from the `model_config` attribute:
        cfg = self.model_config

        feature_dim = cfg.get("feature_dim", 288)

        # Build the inverse model (predicting the action between two observations).
        layers = []
        dense_layers = cfg.get("inverse_net_hiddens", (256,))
        # `in_size` is 2x the feature dim.
        in_size = feature_dim * 2
        for out_size in dense_layers:
            layers.append(nn.Linear(in_size, out_size))
            if cfg.get("inverse_net_activation") not in [None, "linear"]:
                layers.append(
                    get_activation_fn(cfg["inverse_net_activation"], "torch")()
                )
            in_size = out_size
        # Last feature layer of n nodes (action space).
        layers.append(nn.Linear(in_size, self.action_space.n))
        self._inverse_net = nn.Sequential(*layers)

        # Build the forward model (predicting the next observation from current one and
        # action).
        layers = []
        dense_layers = cfg.get("forward_net_hiddens", (256,))
        # `in_size` is the feature dim + action space (one-hot).
        in_size = feature_dim + self.action_space.n
        for out_size in dense_layers:
            layers.append(nn.Linear(in_size, out_size))
            if cfg.get("forward_net_activation") not in [None, "linear"]:
                layers.append(
                    get_activation_fn(cfg["forward_net_activation"], "torch")()
                )
            in_size = out_size
        # Last feature layer of n nodes (feature dimension).
        layers.append(nn.Linear(in_size, feature_dim))
        self._forward_net = nn.Sequential(*layers)

    @override(SelfSupervisedLossAPI)
    def compute_self_supervised_loss(
            self,
            *,
            learner: "TorchLearner",
            module_id: ModuleID,
            config: "AlgorithmConfig",
            batch: Dict[str, Any],
            fwd_out: Dict[str, Any],
    ) -> Dict[str, Any]:
        module = learner.module[module_id].unwrapped()

        # Forward net loss.
        forward_loss = torch.mean(fwd_out[Columns.INTRINSIC_REWARDS])

        # Inverse loss term (predicted action that led from phi to phi' vs
        # actual action taken).
        dist_inputs = module._inverse_net(
            torch.cat([fwd_out["phi"], fwd_out["next_phi"]], dim=-1)
        )
        action_dist = module.get_train_action_dist_cls().from_logits(dist_inputs)

        # Neg log(p); p=probability of observed action given the inverse-NN
        # predicted action distribution.
        inverse_loss = -action_dist.logp(batch[Columns.ACTIONS])
        inverse_loss = torch.mean(inverse_loss)

        # Calculate the ICM loss.
        total_loss = (
                config.learner_config_dict["forward_loss_weight"] * forward_loss
                + (1.0 - config.learner_config_dict["forward_loss_weight"]) * inverse_loss
        )

        learner.metrics.log_dict(
            {
                "mean_intrinsic_rewards": forward_loss,
                "forward_loss": forward_loss,
                "inverse_loss": inverse_loss,
            },
            key=module_id,
            window=1,
        )

        return total_loss
