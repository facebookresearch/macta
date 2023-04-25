# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from rlmeta.core.callbacks import EpisodeCallbacks
from rlmeta.core.types import Action, TimeStep

class MACallbacks(EpisodeCallbacks):
    def __init__(self):
        super().__init__()
    
    def on_episode_start(self, index: int) -> None:
        self.tot_guess = 0
        self.acc_guess = 0
        self.tot_detect = 0
        self.acc_detect = 0

    def on_episode_step(self, index: int, step: int, action: Action,
                        timestep: TimeStep) -> None:
        attacker_info = timestep['attacker'].info
        if attacker_info["is_guess"] and attacker_info['action_mask']['attacker']:
            self.tot_guess += 1
            self.acc_guess += int(attacker_info["guess_correct"])
        detector_info = timestep['detector'].info
        #self.tot_detect += 1
        #self.acc_detect += int(detector_info["guess_correct"])
        if timestep['detector'].done:
            self.tot_detect = 1
            self.acc_detect += int(detector_info["guess_correct"])
            self._custom_metrics["detector_correct_rate"] = self.acc_detect / self.tot_detect
            if attacker_info['action_mask']['attacker']:
                if self.tot_guess>0:
                    self._custom_metrics["attacker_correct_rate"] = self.acc_guess / float(self.tot_guess)
                self._custom_metrics["num_total_guess"] = float(self.tot_guess)
                self._custom_metrics["num_total_attacks"] = float(self.acc_guess)
