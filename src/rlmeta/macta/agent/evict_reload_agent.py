class EvictReloadAgent():

    # the config is the same as the config cor cache_guessing_game_env_impl
    def __init__(self, env_config):
        self.local_step = 0
        self.lat = []
        self.no_prime = False # set to true after first prime
        if "cache_configs" in env_config:
            #self.logger.info('Load config from JSON')
            self.configs = env_config["cache_configs"]
            self.num_ways = self.configs['cache_1']['associativity'] # n-ways in N
            self.cache_size = self.configs['cache_1']['blocks']
            
            # self.n_sets = self.cache_size / self.num_ways # Total number of sets (M) in the cache
        
            attacker_addr_s = env_config["attacker_addr_s"] if "attacker_addr_s" in env_config else 0 # 0 or N*M
            attacker_addr_e = env_config["attacker_addr_e"] if "attacker_addr_e" in env_config else 7
            victim_addr_s = env_config["victim_addr_s"] if "victim_addr_s" in env_config else 0
            victim_addr_e = env_config["victim_addr_e"] if "victim_addr_e" in env_config else 3
            flush_inst = env_config["flush_inst"] if "flush_inst" in env_config else False            
            self.allow_empty_victim_access = env_config["allow_empty_victim_access"] if "allow_empty_victim_access" in env_config else False
            
            # assert(self.num_ways == 1) # currently only support direct-map cache
            # assert(flush_inst == False) # do not allow flush instruction
            #assert((attacker_addr_e - attacker_addr_s) == 2 * (victim_addr_e - victim_addr_s )) # address space must be shared
            #must be no shared address space
            # assert( attacker_addr_s == victim_addr_s)
            # assert( ( attacker_addr_e + 1 == victim_addr_s ) or ( victim_addr_e + 1 == attacker_addr_s ) )
            assert(self.allow_empty_victim_access == False)

    # initialize the agent with an observation
    def observe_init(self, timestep):
        # initialization doing nothing
        self.local_step = 0
        self.lat = []
        self.no_prime = False
        return


    # returns an action
    def act(self, timestep):
        info = {}
        if timestep.observation[0][0][0] == -1:
            #reset the attacker
            #from IPython import embed; embed()
            self.local_step = 0
            self.lat=[]
            self.no_prime = False

        # do evict
        if self.local_step < self.cache_size - ( self.cache_size if self.no_prime else 0 ):#- 1:
            action = self.local_step + ( self.cache_size - (self.cache_size if self.no_prime else 0 )) # do evict
            self.local_step += 1
            return action, info

        elif self.local_step == self.cache_size - (self.cache_size if self.no_prime else 0 ):#- 1: # do victim trigger
            action = self.cache_size # do victim access
            self.local_step += 1
            return action, info

        elif self.local_step < 2 * self.cache_size + 1 -(self.cache_size if self.no_prime else 0 ):#- 1 - 1:# do reload
            action = self.local_step - (self.cache_size if self.no_prime else 0 ) - 1  
            self.local_step += 1
            if action > self.cache_size:
                action += 1
            return action, info

        elif self.local_step == 2 * self.cache_size + 1 - (self.cache_size if self.no_prime else 0 ):# - 1 - 1: # do guess and terminate
            action = 2 * self.cache_size # default assume that last is miss
            for addr in range(1, len(self.lat)):
                if self.lat[addr].int() == 1: # miss
                    action = addr + self.cache_size 
                    break
            self.local_step = 0
            self.lat=[]
            self.no_prime = True
            if action > self.cache_size:
                action+=1
            return action, info
        else:        
            assert(False)
    
    def observe(self, action, timestep):
        if self.local_step < 2 * self.cache_size + 1 + 1 - (self.cache_size if self.no_prime else 0 ) and self.local_step > self.cache_size - (self.cache_size if self.no_prime else 0 ):#- 1:
        ##    self.local_step += 1
            self.lat.append(timestep.observation[0][0])
        return
