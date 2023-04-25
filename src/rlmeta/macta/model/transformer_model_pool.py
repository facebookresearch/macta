# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import os
import sys

from typing import Dict, List, Tuple, Optional, Union

import gym
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import rlmeta.core.remote as remote
import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.ppo.ppo_model import PPOModel
from rlmeta.core.model import DownstreamModel, RemotableModel
from rlmeta.core.server import Server

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from cache_ppo_transformer_model import CachePPOTransformerModel


class CachePPOTransformerModelPool(CachePPOTransformerModel):
    def __init__(self,
                 latency_dim: int,
                 victim_acc_dim: int,
                 action_dim: int,
                 step_dim: int,
                 window_size: int,
                 action_embed_dim: int,
                 step_embed_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 1) -> None:
        super().__init__(latency_dim, victim_acc_dim, action_dim, step_dim,
                         window_size, action_embed_dim, step_embed_dim,
                         hidden_dim, output_dim, num_layers)
        self.history = []
        self.latest = None
        self.use_history = False

    # @remote.remote_method(batch_size=128)
    # def act(self, obs: torch.Tensor, deterministic_policy: torch.Tensor,
    #         reload_model: bool
    #         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     if reload_model:
    #         if self.use_history and len(self.history) > 0:
    #             state_dict = random.choice(self.history)
    #             self.load_state_dict(state_dict)
    #         elif self.latest is not None:
    #             self.load_state_dict(self.latest)
    #         #print("reloading model", reload_model)
    #         #print("length of history:", len(self.history), "use history:", self.use_history, "latest:", self.latest if self.latest is None else len(self.latest))

    @remote.remote_method(batch_size=128)
    def act(
        self, obs: torch.Tensor, deterministic_policy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.use_history and len(self.history) > 0:
            state_dict = random.choice(self.history)
            self.load_state_dict(state_dict)
        elif self.latest is not None:
            self.load_state_dict(self.latest)

        if self._device is None:
            self._device = next(self.parameters()).device

        with torch.no_grad():
            try:
                x = obs.to(self._device)
            except:
                print(obs)
            d = deterministic_policy.to(self._device)
            logpi, v = self.forward(x)

            greedy_action = logpi.argmax(-1, keepdim=True)
            sample_action = logpi.exp().multinomial(1, replacement=True)
            action = torch.where(d, greedy_action, sample_action)
            logpi = logpi.gather(dim=-1, index=action)

            return action.cpu(), logpi.cpu(), v.cpu()

    @remote.remote_method(batch_size=None)
    def push(self, state_dict: Dict[str, torch.Tensor]) -> None:
        # Move state_dict to device before loading.
        # https://github.com/pytorch/pytorch/issues/34880
        device = next(self.parameters()).device
        state_dict = nested_utils.map_nested(lambda x: x.to(device),
                                             state_dict)
        self.latest = state_dict
        self.load_state_dict(state_dict)

    @remote.remote_method(batch_size=None)
    def push_to_history(self, state_dict: Dict[str, torch.Tensor]) -> None:
        # Move state_dict to device before loading.
        # https://github.com/pytorch/pytorch/issues/34880
        device = next(self.parameters()).device
        state_dict = nested_utils.map_nested(lambda x: x.to(device),
                                             state_dict)
        self.latest = state_dict
        self.history.append(self.latest)

    @remote.remote_method(batch_size=None)
    def set_use_history(self, use_history: bool) -> None:
        print("set use history", use_history)
        self.use_history = use_history
        print("after setting:", self.use_history)


class DownstreamModelPool(DownstreamModel):
    def __init__(self,
                 model: nn.Module,
                 server_name: str,
                 server_addr: str,
                 name: Optional[str] = None,
                 timeout: float = 60) -> None:
        super().__init__(model, server_name, server_addr, name, timeout)

    def set_use_history(self, use_history):
        self.client.sync(self.server_name,
                         self.remote_method_name("set_use_history"),
                         use_history)

    def push_to_history(self) -> None:
        state_dict = self.wrapped.state_dict()
        state_dict = nested_utils.map_nested(lambda x: x.cpu(), state_dict)
        self.client.sync(self.server_name,
                         self.remote_method_name("push_to_history"),
                         state_dict)


ModelLike = Union[nn.Module, RemotableModel, DownstreamModel, remote.Remote]


def wrap_downstream_model(model: nn.Module,
                          server: Server,
                          name: Optional[str] = None,
                          timeout: float = 60) -> DownstreamModel:
    return DownstreamModelPool(model, server.name, server.addr, name, timeout)
