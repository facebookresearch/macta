# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

# a textbook prime probe attacker that serve as the agent 
# which can have high reward for the cache guessing game
# used to generate the attack sequence that can be detected by cchunter
# currently it only works for the direct-map cache (associativity=1)
import random
import numpy as np
class PrimeProbeAgent:

    # the config is the same as the config for cache_guessing_game_env_impl
    def __init__(self, env_config):
        self.local_step = 0
        self.lat = []
        self.no_prime = False # set to true after first prime
        if "cache_configs" in env_config:
            #self.logger.info('Load config from JSON')
            self.configs = env_config["cache_configs"]
            self.num_ways = self.configs['cache_1']['associativity'] 
            self.cache_size = self.configs['cache_1']['blocks']
            attacker_addr_s = env_config["attacker_addr_s"] if "attacker_addr_s" in env_config else 4
            attacker_addr_e = env_config["attacker_addr_e"] if "attacker_addr_e" in env_config else 7
            victim_addr_s = env_config["victim_addr_s"] if "victim_addr_s" in env_config else 0
            victim_addr_e = env_config["victim_addr_e"] if "victim_addr_e" in env_config else 3
            flush_inst = env_config["flush_inst"] if "flush_inst" in env_config else False            
            self.allow_empty_victim_access = env_config["allow_empty_victim_access"] if "allow_empty_victim_access" in env_config else False
            
            assert(self.num_ways == 1) # currently only support direct-map cache
            assert(flush_inst == False) # do not allow flush instruction
            assert(attacker_addr_e - attacker_addr_s == victim_addr_e - victim_addr_s ) # address space must be shared
            #must be no shared address space
            assert( ( attacker_addr_e + 1 == victim_addr_s ) or ( victim_addr_e + 1 == attacker_addr_s ) )
            assert(self.allow_empty_victim_access == False)


    def observe_init(self, timestep):
        self.local_step = 0
        self.lat = []
        self.no_prime = False
        return


    def act(self, timestep):
        info = {}
        if timestep.observation[0][0][0] == -1:
            #reset the attacker
            self.local_step = 0
            self.lat=[]
            self.no_prime = False
            self.prime_perm = np.random.permutation(self.cache_size)
            self.probe_perm = np.random.permutation(self.cache_size)
            #self.prime_perm.sort()
            #self.probe_perm.sort()
            
        # do prime
        if self.local_step < self.cache_size -  ( self.cache_size if self.no_prime else 0 ):
            action = self.local_step # do prime
            action = self.prime_perm[action]
            self.local_step += 1
            return action, info

        elif self.local_step == self.cache_size - (self.cache_size if self.no_prime else 0 ):
            # do victim access
            action = self.cache_size
            self.local_step += 1
            return action, info

        elif self.local_step < 2 * self.cache_size + 1 -(self.cache_size if self.no_prime else 0 ):
            # do probe
            action = self.local_step - ( self.cache_size + 1 - (self.cache_size if self.no_prime else 0 ) )
            self.local_step += 1
            action=self.probe_perm[action]
            return action, info

        elif self.local_step == 2 * self.cache_size + 1 - (self.cache_size if self.no_prime else 0 ):
            # do guess and terminate
            # first timestep invisible victim latency, and it is not useful
            action = 2 * self.cache_size
        
            for addr in range(1, len(self.lat)):
                if self.lat[addr].int() == 1: # miss
                    action = self.probe_perm[addr-1] + self.cache_size + 2 
                    break
            self.local_step = 0
            self.lat=[]
            self.no_prime = True
            self.probe_perm = np.random.permutation(self.cache_size)
            return action, info
        else:        
            assert(False)
    
    def observe(self, action, timestep):
        if self.local_step < 2 * self.cache_size + 1 + 1 - (self.cache_size if self.no_prime else 0 ) and self.local_step > self.cache_size - (self.cache_size if self.no_prime else 0 ):
            self.lat.append(timestep.observation[0][0])
        return
