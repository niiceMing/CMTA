# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
"""vae component for the agent."""

from copy import deepcopy
from typing import List, cast

import torch
import torch.nn as nn

from mtrl.agent import utils as agent_utils
from mtrl.agent.components import base as base_component
from mtrl.agent.components import moe_layer
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.utils import config as config_utils
from mtrl.utils.types import ConfigType, ModelType, TensorType
import pdb


def tie_weights(src, trg):
    assert type(src) == type(trg)
    if hasattr(src, "weight"):
        trg.weight = src.weight
    if hasattr(src, "bias"):
        trg.bias = src.bias

class VAEDecoder(nn.Module):
    def __init__(
        self,
        multitask_cfg: ConfigType,
        feature_dim: int,
        num_layers: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        should_tie_decoders: bool = True,
    ):
        """Feedforward encoder for state observations.

        Args:
            env_obs_shape (List[int]): shape of the observation that the actor gets.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
            feature_dim (int): feature dimension.
            num_layers (int, optional): number of layers. Defaults to 2.
            hidden_dim (int, optional): number of conv filters per layer. Defaults to 32.
            should_tie_decoders (bool): should the feed-forward layers be tied.
        """

        super().__init__()

        self.num_layers = num_layers
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_var = nn.Linear(feature_dim, latent_dim)

        self.decoder = agent_utils.build_mlp(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=feature_dim,
        )
        self.should_tie_decoders = should_tie_decoders

    def encode(self, encoding: TensorType) -> List[TensorType]:
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(encoding)
        log_var = self.fc_var(encoding)

        return [mu, log_var]

    def reparameterize(self, mu: TensorType, logvar: TensorType) -> TensorType:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu    

    def forward(self, encoding: TensorType, detach: bool = False):
        mu, log_var = self.encode(encoding)
        z = self.reparameterize(mu, log_var)
        output = self.decoder(z)
        if detach:
            output = output.detach()
        return  output, mu, log_var
        
    def copy_conv_weights_from(self, source):
        if self.should_tie_decoders:
            tie_weights(src=source.fc_mu, trg=self.fc_mu)
            tie_weights(src=source.fc_var, trg=self.fc_var)
            for src, trg in zip(source.decoder, self.decoder):  # type: ignore[call-overload]
                tie_weights(src=src, trg=trg)


_AVAILABLE_ENCODERS = {
    "vae": VAEDecoder,
}


def make_decoder(
    encoder_cfg: ConfigType,
    multitask_cfg: ConfigType,
    # device: Optional[torch.device] = None,
):
    key = "type_to_select"
    if key in encoder_cfg:
        encoder_type_to_select = encoder_cfg[key]
        encoder_cfg = encoder_cfg[encoder_type_to_select]

    if encoder_cfg.type in ["moe", "fmoe"]:
        feature_dim = encoder_cfg.encoder_cfg.feature_dim
    else:
        feature_dim = encoder_cfg.feature_dim



    # cfg_to_use = config_utils.make_config_mutable(
    #     config_utils.unset_struct(deepcopy(encoder_cfg))
    # )
    # cfg_to_use.pop("type")
    return _AVAILABLE_ENCODERS["vae"](
        multitask_cfg=multitask_cfg,
        feature_dim = feature_dim,
    )


