# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
"""Actor component for the agent."""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mtrl.agent import utils as agent_utils
from mtrl.agent.components import base as base_component
from mtrl.agent.components import encoder, moe_layer
from mtrl.agent.components.soft_modularization import SoftModularizedMLP
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.agent.ds.task_info import TaskInfo
from mtrl.utils.types import ConfigType, ModelType, TensorType
from mtrl.env.types import ObsType
import pdb

class Classifier(nn.Module):
    def __init__(
        self,
        env_obs_shape: List[int],
        num_env: int,
        num_layers = 3,
        hidden_dim = 128,
    ):
        """Feedforward encoder for state observations.

        Args:
            env_obs_shape (List[int]): shape of the observation that the actor gets.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
            feature_dim (int): feature dimension.
            num_layers (int, optional): number of layers. Defaults to 2.
            hidden_dim (int, optional): number of conv filters per layer. Defaults to 32.
        """

        super().__init__()

        assert len(env_obs_shape) == 1

        self.num_layers = num_layers
        self.trunk = agent_utils.build_mlp(
            input_dim=env_obs_shape[0],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=num_env,
        )

        self.apply(agent_utils.weight_init)


    def forward(self, env_obs):
        x = self.trunk(env_obs)
        # pdb.set_trace()
        # pred = torch.nn.functional.softmax(x, dim =1)
        return x
