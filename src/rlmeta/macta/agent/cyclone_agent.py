# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import copy

from typing import Any, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import gym

from sklearn import svm
from sklearn.model_selection import cross_val_score
import pickle
import os

class CycloneAgent:
    def __init__(self,
                 env_config,
                 svm_model_path=None,
                 keep_latency=True,
                 mode='data') -> None:
        """
        mode:
            if mode is 'data': collect data, this agent will not act
            if mode is 'active': this agent will act
        """
        self.episode_length = env_config.get("episode_length", 64)
        self.keep_latency = keep_latency
        self.mode = mode # 'data' or 'active'
        self.env_config = env_config
        if svm_model_path is not None:
            print("loading cyclone agent from ", svm_model_path)
            self.clf = pickle.load(open(svm_model_path,'rb'))
        
        self.cyclone_window_size = env_config.get("cyclone_window_size", 4)
        self.cyclone_interval_size = env_config.get("cyclone_interval_size", 17) #was 16
        self.cyclone_num_buckets = env_config.get("cyclone_num_buckets", 8) #was 4
        self.cyclone_bucket_size = env_config.cache_configs.cache_1.blocks / self.cyclone_num_buckets

        self.observe_init(timestep=None)

    def cyclone_detect(self, cyclone_counters):
        """
        if the loaded classifier is 
            svm: model will output 1 for attacker and 0 for benign
            oneclass svm: model will output -1 for attacker and 1 for benign
        """
        x = np.array(cyclone_counters).reshape(-1)
        y = self.clf.predict([x])[0]
        for step in self.clf.steps:
            if "oneclasssvm" in step:
                y = int(-y/2+0.5)
                break
        return y

    def observe_init(self, timestep):
        self.local_step = 0
        self.cyclone_counters = []
        for j in range(self.cyclone_num_buckets):
            temp =[]
            for i in range(self.cyclone_window_size):
                temp.append(0)
            self.cyclone_counters.append(temp)

    def act(self, timestep):
        if timestep.observation[0][0][0] == -1:
            #reset the observation
            self.observe_init(timestep=timestep)

        cur_step_obs = timestep.observation[0][0]
        self.local_step = max(cur_step_obs[-1],0)
        info = timestep.info
        if "cyclic_set_index" in info.keys() and info["cyclic_set_index"] != -1: 
            cyclic_set_index = int(info["cyclic_set_index"])
            if self.local_step < self.episode_length:
                self.cyclone_counters[int(cyclic_set_index / self.cyclone_bucket_size)][int((self.local_step-1) / self.cyclone_interval_size)] += 1

        if timestep.observation[0][0][-1] >= self.episode_length-1 and self.mode=='active': 
            action = self.cyclone_detect(self.cyclone_counters)
        else:
            action = 0
        return action, info


    def observe(self, action, timestep):
        return


if __name__ == "__main__":
    CycloneAgent(env_config={})
