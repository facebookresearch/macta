# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

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
from rlmeta.agents.ppo.ppo_agent import PPOAgent

console = Console()


class PPOAgent(PPOAgent):

    def __init__(self,
                 model: ModelLike,
                 deterministic_policy: bool = False,
                 replay_buffer: Optional[ReplayBufferLike] = None,
                 controller: Optional[ControllerLike] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 batch_size: int = 128,
                 grad_clip: float = 1.0,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 eps_clip: float = 0.2,
                 vf_loss_coeff: float = 0.5,
                 entropy_coeff: float = 0.01,
                 dual_clip: Optional[float] = None,
                 reward_rescaling: bool = True,
                 advantage_normalization: bool = True,
                 value_clip: bool = True,
                 learning_starts: Optional[int] = None,
                 push_every_n_steps: int = 1) -> None:
        super().__init__(model, deterministic_policy, replay_buffer,
                         controller, optimizer, batch_size, grad_clip,
                         gamma, gae_lambda, eps_clip, vf_loss_coeff,
                         entropy_coeff, reward_rescaling,
                         advantage_normalization, value_clip,
                         learning_starts, push_every_n_steps)

        self.dual_clip = dual_clip

    def set_use_history(self, use_history):
        self.model.set_use_history(use_history)

    def act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        # reload_model = timestep.info.get("episode_reset", False)
        # action, logpi, v = self.model.act(
        #     obs, torch.tensor([self.deterministic_policy]), reload_model)
        action, logpi, v = self.model.act(
            obs, torch.tensor([self.deterministic_policy]))
        return Action(action, info={"logpi": logpi, "v": v})

    async def async_act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        # reload_model = timestep.info.get("episode_reset", False)
        # action, logpi, v = await self.model.async_act(
        #     obs, torch.tensor([self.deterministic_policy]), reload_model)
        action, logpi, v = await self.model.async_act(
            obs, torch.tensor([self.deterministic_policy]))
        return Action(action, info={"logpi": logpi, "v": v})

    def train(self, num_steps: int) -> Optional[StatsDict]:
        #self.controller.set_phase(Phase.TRAIN, reset=True)

        self.replay_buffer.warm_up(self.learning_starts)
        stats = StatsDict()

        console.log(f"Training for num_steps = {num_steps}")
        for step in track(range(num_steps), description="Training..."):
            t0 = time.perf_counter()
            batch = self.replay_buffer.sample(self.batch_size)
            t1 = time.perf_counter()
            step_stats = self._train_step(batch)
            t2 = time.perf_counter()
            time_stats = {
                "sample_data_time/ms": (t1 - t0) * 1000.0,
                "batch_learn_time/ms": (t2 - t1) * 1000.0,
            }
            stats.extend(step_stats)
            stats.extend(time_stats)

            if step % self.push_every_n_steps == self.push_every_n_steps - 1:
                self.model.push()

        #push the last checkpoint to the model history checkpoint
        #try:
        #    self.model.push_to_history()
        #except:
        #    pass
        episode_stats = self.controller.get_stats()
        stats.update(episode_stats)

        return stats

    def eval(self, num_episodes: Optional[int] = None) -> Optional[StatsDict]:
        #self.controller.set_phase(Phase.EVAL, limit=num_episodes, reset=True)
        while self.controller.get_count() < num_episodes:
            time.sleep(1)
        stats = self.controller.get_stats()
        return stats

    def _policy_loss(self, logpi: torch.Tensor, old_logpi: torch.Tensor,
                     adv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.advantage_normalization:
            # Advantage normalization
            std, mean = torch.std_mean(adv, unbiased=False)
            adv = (adv - mean) / (std + 1e-8)

        ratio = (logpi - old_logpi).exp()

        # Dual-clip PPO.
        # https://arxiv.org/pdf/1912.09729.pdf
        if self.dual_clip is not None:
            # Assume self.dual_clip > 1 + self.eps_clip.
            # If adv > 0, ratio > self.dual_clip -> raito > 1 + self.eps_clip.
            # If adv < 0, ratio will be dual clipped.
            ratio = ratio.clamp_max(self.dual_clip)

        # Policy clip
        ratio_clamp = ratio.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip)
        surr1 = ratio * adv
        surr2 = ratio_clamp * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        return policy_loss, ratio
