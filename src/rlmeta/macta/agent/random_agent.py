# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from rich.console import Console
from rich.progress import track

import rlmeta.utils.data_utils as data_utils
import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.agent import Agent
from rlmeta.core.controller import Controller, ControllerLike, Phase
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.rescalers import Rescaler, RMSRescaler
from rlmeta.core.types import Action, TimeStep
from rlmeta.core.types import Tensor, NestedTensor
from rlmeta.utils.stats_dict import StatsDict

import random

console = Console()


class RandomAgent(Agent):

    def __init__(self,
                action_space):
        super().__init__()
        self.action_space = action_space

    def act(self, timestep: TimeStep) -> Action:
        action = random.randint(0, self.action_space-1)
        return Action(action, {})

    async def async_act(self, timestep: TimeStep) -> Action:
        action = random.randint(0, self.action_space-1)

        return Action(action, {})

    async def async_observe_init(self, timestep: TimeStep) -> None:
        pass

    async def async_observe(self, action: Action,
                            next_timestep: TimeStep) -> None:
        pass

    def update(self) -> None:
        pass
    
    async def async_update(self) -> None:
        pass



