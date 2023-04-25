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

class CachePPOTransformerModel(PPOModel):
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
        super().__init__()

        self.latency_dim = latency_dim
        self.victim_acc_dim = victim_acc_dim
        self.action_dim = action_dim
        self.step_dim = step_dim
        self.window_size = window_size

        self.action_embed_dim = action_embed_dim
        self.step_embed_dim = step_embed_dim
        self.input_dim = (self.latency_dim + self.victim_acc_dim +
                          self.action_embed_dim + self.step_embed_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.action_embed = nn.Embedding(self.action_dim,
                                         self.action_embed_dim)
        self.step_embed = nn.Embedding(self.step_dim, self.step_embed_dim)

        self.linear_i = nn.Linear(self.input_dim, self.hidden_dim)
        # self.linear_o = nn.Linear(self.hidden_dim * self.window_size,
        #                           self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim,
                                                   nhead=8,
                                                   dropout=0.0)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

        self.linear_a = nn.Linear(self.hidden_dim, self.output_dim)
        self.linear_v = nn.Linear(self.hidden_dim, 1)

        self._device = None

    def make_one_hot(self, src: torch.Tensor,
                     num_classes: int) -> torch.Tensor:
        mask = (src == -1)
        src = src.masked_fill(mask, 0)
        ret = F.one_hot(src, num_classes)
        return ret.masked_fill(mask.unsqueeze(-1), 0.0)

    def make_embedding(self, src: torch.Tensor,
                       embed: nn.Embedding) -> torch.Tensor:
        mask = (src == -1)
        src = src.masked_fill(mask, 0)
        ret = embed(src)
        return ret.masked_fill(mask.unsqueeze(-1), 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = obs.to(torch.int64)
        assert obs.dim() == 3

        # batch_size = obs.size(0)
        l, v, act, stp = torch.unbind(obs, dim=-1)
        l = self.make_one_hot(l, self.latency_dim)
        v = self.make_one_hot(v, self.victim_acc_dim)
        act = self.make_embedding(act, self.action_embed)
        stp = self.make_embedding(stp, self.step_embed)

        x = torch.cat((l, v, act, stp), dim=-1)
        x = self.linear_i(x)
        x = x.transpose(0, 1).contiguous()
        h = self.encoder(x)
        # h = self.linear_o(h.view(batch_size, -1))
        h = h.mean(dim=0)

        p = self.linear_a(h)
        logpi = F.log_softmax(p, dim=-1)
        v = self.linear_v(h)

        return logpi, v

    @remote.remote_method(batch_size=128)
    def act(
        self, obs: torch.Tensor, deterministic_policy: torch.Tensor, reload_model=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._device is None:
            self._device = next(self.parameters()).device

        with torch.no_grad():
            #print(obs)
            x = obs.to(self._device)
            d = deterministic_policy.to(self._device)
            logpi, v = self.forward(x)

            greedy_action = logpi.argmax(-1, keepdim=True)
            sample_action = logpi.exp().multinomial(1, replacement=True)
            action = torch.where(d, greedy_action, sample_action)
            logpi = logpi.gather(dim=-1, index=action)

            return action.cpu(), logpi.cpu(), v.cpu()


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
        super().__init__(latency_dim,
                        victim_acc_dim,
                        action_dim,
                        step_dim,
                        window_size,
                        action_embed_dim,
                        step_embed_dim,
                        hidden_dim,
                        output_dim,
                        num_layers)
        self.history = []
        self.latest = None
        self.use_history = False

    @remote.remote_method(batch_size=128)
    def act(
        self, 
        obs: torch.Tensor, 
        deterministic_policy: torch.Tensor,
        reload_model: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if reload_model:
            if self.use_history and len(self.history)>0:
                state_dict = random.choice(self.history)
                self.load_state_dict(state_dict)
            elif self.latest is not None:
                self.load_state_dict(self.latest)
            #print("reloading model", reload_model)
            #print("length of history:", len(self.history), "use history:", self.use_history, "latest:", self.latest if self.latest is None else len(self.latest))
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
        state_dict = nested_utils.map_nested(lambda x: x.to(device), state_dict)
        self.latest = state_dict
        self.load_state_dict(state_dict)
    
    @remote.remote_method(batch_size=None)
    def push_to_history(self, state_dict: Dict[str, torch.Tensor]) -> None:
        # Move state_dict to device before loading.
        # https://github.com/pytorch/pytorch/issues/34880
        device = next(self.parameters()).device
        state_dict = nested_utils.map_nested(lambda x: x.to(device), state_dict)
        self.latest = state_dict
        self.history.append(self.latest)
   
    @remote.remote_method(batch_size=None) 
    def set_use_history(self, use_history:bool) -> None:
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
        self.client.sync(self.server_name, self.remote_method_name("set_use_history"),
                         use_history)
    
    def push_to_history(self) -> None:
        state_dict = self.wrapped.state_dict()
        state_dict = nested_utils.map_nested(lambda x: x.cpu(), state_dict)
        self.client.sync(self.server_name, self.remote_method_name("push_to_history"),
                         state_dict)    

ModelLike = Union[nn.Module, RemotableModel, DownstreamModel, remote.Remote]


def wrap_downstream_model(model: nn.Module,
                          server: Server,
                          name: Optional[str] = None,
                          timeout: float = 60) -> DownstreamModel:
    return DownstreamModelPool(model, server.name, server.addr, name, timeout)
