import os
import copy
import sys

from typing import Any, Dict, Sequence, Tuple
from collections import deque

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import hydra
import gym

from .cache_guessing_game_env import CacheGuessingGameEnv


class CacheAttackerDetectorEnv(gym.Env):
    def __init__(
        self,
        env_config: Dict[str, Any],
        keep_latency: bool = True,
    ) -> None:
        #env_config["cache_state_reset"] = False

        self.reset_observation = env_config.get("reset_observation", False)
        self.keep_latency = keep_latency
        self.env_config = env_config
        self.episode_length = env_config.get("episode_length", 80)
        self.threshold = env_config.get("threshold", 0.8)

        self._env = CacheGuessingGameEnv(env_config)
        self.validation_env = CacheGuessingGameEnv(env_config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        self.victim_address_min = self._env.victim_address_min
        self.victim_address_max = self._env.victim_address_max
        self.attacker_address_max = self._env.attacker_address_max
        self.attacker_address_min = self._env.attacker_address_min
        self.victim_address = self._env.victim_address
        self.opponent_weights = env_config.get("opponent_weights", [0.5, 0.5])
        self.opponent_agent = random.choices(['benign', 'attacker'],
                                             weights=self.opponent_weights,
                                             k=1)[0]
        self.action_mask = {
            'detector': True,
            'attacker': self.opponent_agent == 'attacker',
            'benign': self.opponent_agent == 'benign'
        }
        self.step_count = 0
        self.max_step = 64
        self.detector_obs = deque([[-1, -1, -1, -1]] * self.max_step)
        self.random_domain = random.choice([0, 1])
        self.detector_reward_scale = 0.1  #1.0

    def reset(self, victim_address=-1):
        """
        returned obs = { agent_name : obs }
        """
        self.opponent_agent = random.choices(['benign', 'attacker'],
                                             weights=self.opponent_weights,
                                             k=1)[0]
        self.action_mask = {
            'detector': True,
            'attacker': self.opponent_agent == 'attacker',
            'benign': self.opponent_agent == 'benign'
        }
        self.step_count = 0
        opponent_obs = self._env.reset(victim_address=victim_address,
                                       reset_cache_state=True)
        self.victim_address = self._env.victim_address
        self.detector_obs = deque([[-1, -1, -1, -1]] * self.max_step)
        self.random_domain = random.choice([0, 1])
        obs = {}
        obs['detector'] = np.array(list(reversed(self.detector_obs)))
        obs['attacker'] = opponent_obs
        obs['benign'] = opponent_obs
        return obs

    def get_detector_obs(self, opponent_obs, opponent_info):
        cur_opponent_obs = copy.deepcopy(opponent_obs[0])
        if not np.any(cur_opponent_obs == -1):
            # Make sure the observation does not leak information for detector
            # TODO should we include step number? - yes we should - the guess step should not be observed by the detector
            # attacker obs: r, victim_accessed, original action, current step
            # detector obs: r, domain_id, memory address, 0
            if opponent_info.get('invoke_victim'):
                cur_opponent_obs[0] = opponent_info['victim_latency']
                cur_opponent_obs[1] = self.random_domain  #1
                cur_opponent_obs[2] = opponent_info['victim_address']
            else:
                cur_opponent_obs[1] = 1 - self.random_domain  #0
                cur_opponent_obs[2] = opponent_info['attacker_address']
            cur_opponent_obs[3] = self.step_count  #0#self.step_count
            self.detector_obs.append(cur_opponent_obs)
            self.detector_obs.popleft()
        return np.array(list(reversed(self.detector_obs)))

    def compute_reward(self,
                       action,
                       reward,
                       opponent_done,
                       opponent_attack_success=False):
        action_detector = action['detector']
        action_attacker = action['attacker']

        # determine detector's reward
        detector_flag = False
        detector_correct = False

        if action_detector == 1:
            detector_flag = True
            # terminate the episode
            # detector flag the opponent as an attacker
            if self.opponent_agent == 'benign':
                #detector_reward = - self.max_step + self.step_count - 1 #punish false positive
                detector_reward = -10 * self.max_step
            elif self.opponent_agent == 'attacker':
                detector_reward = max(self.max_step - self.step_count, 0)
                detector_correct = True
        else:
            # else receive a timestep penalty
            if self.opponent_agent == 'benign':
                detector_reward = 0  # no reward if benign agent
                detector_correct = True
            else:
                detector_reward = 0  # no reward if attacker has done nothing meaningful

        if action_detector == 0 and self.opponent_agent == 'attacker' and opponent_done and opponent_attack_success:
            # attacker episode terminate, and has attacked *successfully*
            detector_reward = -20
        elif action_detector == 0 and self.opponent_agent == 'attacker' and opponent_done:
            # attacker episode terminates, but has done nothing successfully
            detector_reward = 0

        attacker_reward = reward['attacker']

        if action_detector == 1 and detector_correct:
            # the attacker should not receive as much reward if being detected
            attacker_reward -= 10
        #else:
        #    attacker_reward += 0.1

        rew = {}
        rew['detector'] = detector_reward * self.detector_reward_scale
        rew['attacker'] = attacker_reward

        info = {}
        info['guess_correct'] = detector_correct
        return rew, info

    def step(self, action):
        self.step_count += 1
        #if self.step_count <=8: action['detector']=0
        obs = {}
        reward = {}
        done = {'__all__': False}
        info = {}
        # Attacker update
        action_info = action.get('info')
        #if self.opponent_agent == 'benign':
        if action_info:
            benign_reset_victim = action_info.get('reset_victim_addr', False)
            benign_victim_addr = action_info.get('victim_addr', None)
            if self.opponent_agent == 'benign' and benign_reset_victim:
                self._env.set_victim(benign_victim_addr)
                self.victim_address = self._env.victim_address
        opponent_obs, opponent_reward, opponent_done, opponent_info = self._env.step(
            action[self.opponent_agent])
        if opponent_done:
            opponent_obs = self._env.reset(reset_cache_state=True)
            self.victim_address = self._env.victim_address
            self.step_count -= 1  # The reset/guess step should not be counted
        if self.step_count >= self.max_step:
            detector_done = True
        else:
            detector_done = False
        if action["detector"] == 1:  # Raise Hard Flag
            detector_done = True  # Terminate the episode
        # attacker
        obs['attacker'] = opponent_obs
        reward['attacker'] = opponent_reward
        done['attacker'] = detector_done  #Figure out correctness
        info['attacker'] = opponent_info

        #benign
        obs['benign'] = opponent_obs
        reward['benign'] = opponent_reward
        done['benign'] = detector_done  #Figure out correctness
        info['benign'] = opponent_info
        opponent_attack_success = opponent_info.get('guess_correct', False)

        # obs, reward, done, info
        updated_reward, updated_info = self.compute_reward(
            action, reward, opponent_done, opponent_attack_success)
        reward['attacker'] = updated_reward['attacker']
        reward['detector'] = updated_reward['detector']
        obs['detector'] = self.get_detector_obs(opponent_obs, opponent_info)
        done['detector'] = detector_done
        info['detector'] = {
            "guess_correct": updated_info["guess_correct"],
            "is_guess": bool(action['detector'])
        }
        info['detector'].update(opponent_info)
        # Change the criteria to determine wether the game is done
        if detector_done:
            done['__all__'] = True
        #from IPython import embed; embed()

        info['__all__'] = {'action_mask': self.action_mask}

        for k, v in info.items():
            info[k].update({'action_mask': self.action_mask})
        #print(obs["detector"])
        return obs, reward, done, info


@hydra.main(config_path="../config", config_name="macta_ppo")
def main(cfg):
    env = CacheAttackerDetectorEnv(cfg.env_config)
    env.opponent_weights = [0, 1]
    action_space = env.action_space
    obs = env.reset()
    done = {'__all__': False}
    prev_a = 10
    for k in range(2):
        i = 0
        while not done['__all__']:
            i += 1
            print("step: ", i)
            action = {
                'attacker': 9 if (prev_a == 10 or i < 64) else
                10,  #np.random.randint(low=9, high=11),
                'benign': np.random.randint(low=2, high=5),
                'detector': np.random.randint(low=0, high=1)
            }
            prev_a = action['attacker']
            obs, reward, done, info = env.step(action)
            #print("obs: ", obs['detector'])
            print("action: ", action)
            print("victim: ", env.victim_address, env._env.victim_address)

            #print("done:", done)
            print("reward:", reward)
            #print(env.victim_address_min, env.victim_address_max)
            #print("info:", info )
            if info['attacker'].get('invoke_victim') or info['attacker'].get(
                    'is_guess') == True:
                print(info['attacker'])
        obs = env.reset()
        done = {'__all__': False}


if __name__ == "__main__":
    main()
