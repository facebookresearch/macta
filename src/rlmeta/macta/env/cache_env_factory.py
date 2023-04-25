# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import os
import sys

from typing import Any, Dict

from rlmeta.envs.env import Env, EnvFactory
from rlmeta.envs.gym_wrappers import GymWrapper

from .cache_guessing_game_env import CacheGuessingGameEnv


class CacheEnvWrapperFactory(EnvFactory):
    def __init__(self, env_config: Dict[str, Any]) -> None:
        self._env_config = env_config

    @property
    def env_config(self) -> Dict[str, Any]:
        return self._env_config

    def __call__(self, index: int, *args, **kwargs) -> Env:
        env = CacheGuessingGameEnv(self.env_config)
        env = GymWrapper(env)
        return env
