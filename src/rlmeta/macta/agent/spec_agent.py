# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import sys
import time

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

import torch
import torch.nn as nn

from rich.console import Console
from rich.progress import track

import rlmeta.utils.data_utils as data_utils
import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.agent import Agent, AgentFactory
from rlmeta.core.controller import Controller, ControllerLike, Phase
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.rescalers import Rescaler, RMSRescaler
from rlmeta.core.types import Action, TimeStep
from rlmeta.core.types import Tensor, NestedTensor
from rlmeta.utils.stats_dict import StatsDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.trace_parser import load_trace

console = Console()


class SpecAgent(Agent):
    def __init__(self, env_config, trace, legacy_trace_format: bool = False):
        super().__init__()

        self.local_step = 0
        self.lat = []
        self.no_prime = False  # set to true after first prime

        if "cache_configs" in env_config:
            #self.logger.info('Load config from JSON')
            self.configs = env_config["cache_configs"]
            self.num_ways = self.configs['cache_1']['associativity']
            self.cache_size = self.configs['cache_1']['blocks']

            # attacker_addr_s = env_config[
            #     "attacker_addr_s"] if "attacker_addr_s" in env_config else 4
            # attacker_addr_e = env_config[
            #     "attacker_addr_e"] if "attacker_addr_e" in env_config else 7
            # victim_addr_s = env_config[
            #     "victim_addr_s"] if "victim_addr_s" in env_config else 0
            # victim_addr_e = env_config[
            #     "victim_addr_e"] if "victim_addr_e" in env_config else 3
            # flush_inst = env_config[
            #     "flush_inst"] if "flush_inst" in env_config else False
            # self.allow_empty_victim_access = env_config[
            #     "allow_empty_victim_access"] if "allow_empty_victim_access" in env_config else False

            attacker_addr_s = env_config.get("attacker_addr_s", 4)
            attacker_addr_e = env_config.get("attacker_addr_e", 7)
            victim_addr_s = env_config.get("victim_addr_s", 0)
            victim_addr_e = env_config.get("victim_addr_e", 3)
            flush_inst = env_config.get("flush_inst", False)

            self.allow_empty_victim_access = env_config.get(
                "allow_empty_victim_access", False)

            assert (self.num_ways == 1
                    )  # currently only support direct-map cache
            assert (flush_inst == False)  # do not allow flush instruction
            assert (attacker_addr_e - attacker_addr_s == victim_addr_e -
                    victim_addr_s)  # address space must be shared
            #must be no shared address space
            assert ((attacker_addr_e + 1 == victim_addr_s)
                    or (victim_addr_e + 1 == attacker_addr_s))
            assert (self.allow_empty_victim_access == False)

        # self.cache_line_size = 8  #TODO: remove the hardcode
        self.cache_line_size = env_config.get("cache_line_size", 64)
        self.trace = trace
        self.trace_length = (len(self.trace)
                             if legacy_trace_format else self.trace.shape[0])
        assert isinstance(self.trace,
                          (list if legacy_trace_format else np.ndarray))
        self.legacy_trace_format = legacy_trace_format

        # self.trace_length = len(self.trace)
        # line = self.trace[0]
        # self.domain_id_0 = line[0]
        # self.domain_id_1 = line[0]
        # local_step = 0
        # while len(line) > 0:
        #     local_step += 1
        #     line = self.trace[local_step]
        #     self.domain_id_1 = line[0]
        #     if self.domain_id_1 != self.domain_id_0:
        #         break
        self._get_domain_ids()
        assert isinstance(self.domain_id_0, (int, np.int64))
        assert isinstance(self.domain_id_1, (int, np.int64))

        self.start_idx = random.randint(0, self.trace_length - 1)
        self.step = 0

        # print(f"[Agent] cache_line_size = {self.cache_line_size}")

    def act(self, timestep: TimeStep) -> Action:
        # line = self.trace[(self.start_idx + self.step) % self.trace_length]
        # if self.step >= self.trace_length:
        #     self.step = 0
        # else:
        #     self.step += 1
        # if len(line) == 0:
        #     action = self.cache_size
        #     addr = 0  #addr % self.cache_size
        #     info = {"file_done": True}
        #     return Action(action, info)
        # domain_id = line[0]
        # addr = int(int(line[3], 16) / self.cache_line_size)

        idx = (self.start_idx + self.step) % self.trace_length
        self.step = (self.step + 1) % self.trace_length

        if self.legacy_trace_format:
            line = self.trace[idx]
            domain_id = int(line[0])
            addr = int(line[3], 16) // self.cache_line_size
        else:
            domain_id, addr = self.trace[idx]
            addr //= self.cache_line_size

        assert isinstance(domain_id, (int, np.int64))
        assert isinstance(addr, (int, np.int64))

        action = addr % self.cache_size
        if domain_id == self.domain_id_0:  # attacker access
            action = addr % self.cache_size
            info = {}
        else:  # domain_id = self.domain_id_1: # victim access
            action = self.cache_size
            addr = addr % self.cache_size
            info = {"reset_victim_addr": True, "victim_addr": addr}
        return Action(action, info)

    async def async_act(self, timestep: TimeStep) -> Action:
        return self.act(timestep)

    async def async_observe_init(self, timestep: TimeStep) -> None:
        pass

    async def async_observe(self, action: Action,
                            next_timestep: TimeStep) -> None:
        pass

    def update(self) -> None:
        pass

    async def async_update(self) -> None:
        pass

    def _get_domain_ids(self) -> None:
        self.domain_id_0 = (int(self.trace[0][0])
                            if self.legacy_trace_format else self.trace[0, 0])
        self.domain_id_1 = (int(self.trace[0][0])
                            if self.legacy_trace_format else self.trace[0, 0])
        for i in range(self.trace_length):
            cur = (int(self.trace[i][0])
                   if self.legacy_trace_format else self.trace[i, 0])
            if cur != self.domain_id_0:
                self.domain_id_1 = cur
                break


class SpecAgentFactory(AgentFactory):
    def __init__(self,
                 env_config: Dict[str, Any],
                 trace_files: Sequence[str],
                 trace_limit: int,
                 legacy_trace_format: bool = False) -> None:
        self.env_config = env_config
        self.trace_files = trace_files
        self.trace_limit = trace_limit
        self.legacy_trace_format = legacy_trace_format

    def __call__(self, index: int) -> SpecAgent:
        spec_trace = self._load_trace(index)
        return SpecAgent(self.env_config,
                         spec_trace,
                         legacy_trace_format=self.legacy_trace_format)

    def _load_trace(self, index: int) -> np.ndarray:
        trace_file = self.trace_files[index % len(self.trace_files)]

        print(f"[SpecAgentFactory] agent [{index}] load {trace_file}")

        spec_trace = load_trace(trace_file,
                                limit=self.trace_limit,
                                legacy_trace_format=self.legacy_trace_format)
        return spec_trace
