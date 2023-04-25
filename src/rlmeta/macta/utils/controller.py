# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import IntEnum
from typing import Dict, Optional, Union

import rlmeta.core.remote as remote

from rlmeta.utils.stats_dict import StatsDict


class Phase(IntEnum):
    NONE = 0
    TRAIN = 1
    EVAL = 2
    TRAIN_ATTACKER = 3
    TRAIN_DETECTOR = 4
    EVAL_ATTACKER = 5
    EVAL_DETECTOR = 6
    


class Controller(remote.Remotable):

    def __init__(self, identifier: Optional[str] = None) -> None:
        super().__init__(identifier)
        self._phase = Phase.NONE
        self._count = 0
        self._limit = None
        self._stats = StatsDict()

    def __repr__(self):
        return f"Controller(phase={self._phase})"

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def count(self) -> int:
        return self._count

    @property
    def stats(self) -> StatsDict:
        return self._stats

    @remote.remote_method(batch_size=None)
    def reset(self) -> None:
        self.set_phase(Phase.NONE, reset=True)

    @remote.remote_method(batch_size=None)
    def get_phase(self) -> Phase:
        return self.phase

    @remote.remote_method(batch_size=None)
    def set_phase(self,
                  phase: Phase,
                  limit: Optional[int] = None,
                  reset: bool = False) -> None:
        self._phase = phase
        self._limit = limit
        if reset:
            self._count = 0
            self._stats.reset()

    @remote.remote_method(batch_size=None)
    def get_count(self) -> int:
        return self.count

    @remote.remote_method(batch_size=None)
    def get_stats(self) -> StatsDict:
        return self.stats

    @remote.remote_method(batch_size=None)
    def add_episode(self, phase: Phase, stats: Dict[str, float]) -> None:
        if phase == self._phase and (self._limit is None or
                                     self._count < self._limit):
            self._count += 1
            self._stats.extend(stats)

class DummyController(Controller):

    def __init__(self, identifier: Optional[str] = None) -> None:
        super().__init__(identifier)
    
    @remote.remote_method(batch_size=None)
    def set_phase(self,
                  phase: Phase,
                  limit: Optional[int] = None,
                  reset: bool = False) -> None:
        pass

ControllerLike = Union[Controller, remote.Remote, DummyController]
