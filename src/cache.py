import math, block, response
import pprint
from replacement_policy import * 

class Cache:
    def __init__(self, name, word_size, block_size, n_blocks, associativity, hit_time, write_time, write_back, logger, next_level=None, rep_policy='', prefetcher="none", verbose=False):
        #Parameters configured by the user
        self.name = name
        self.word_size = word_size
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.associativity = associativity
        self.hit_time = hit_time
        self.cflush_time = hit_time # assume flush is as fast as hit since
        self.write_time = write_time
        self.write_back = write_back
        self.logger = logger
        self.logger.disabled = False#True
        self.set_rep_policy = {}
        self.verbose = verbose
        if rep_policy == 'lru':
            self.vprint("use lru") 
            self.rep_policy = lru_policy
        elif rep_policy == 'tree_plru':
            self.vprint("use tree_plru") 
            self.rep_policy = tree_plru_policy
        elif rep_policy == 'rand':
            self.vprint("use rand") 
            self.rep_policy = rand_policy
        elif rep_policy == 'plru_pl':
            self.vprint("use plru_pl") 
            self.rep_policy = plru_pl_policy
        elif rep_policy == 'brrip':
            self.vprint("use brrip")
            #assert(False) 
            self.rep_policy = brrip_policy
        else:
            self.rep_policy = lru_policy
            if name == 'cache_1':
                self.vprint("no rep_policy specified or policy specified not exist")
                self.vprint("use lru_policy")
                #assert(False)

        self.vprint("use " + prefetcher + "  prefetcher")

        # prefetcher == "none" "nextline" "stream"
        self.prefetcher = prefetcher
        if self.prefetcher == "stream":
            self.prefetcher_table =[]
            self.num_prefetcher_entry = 2
            for i in range(self.num_prefetcher_entry):
                temp = {"first": -1, "second": -1}
                self.prefetcher_table.append(temp)

        #Total number of sets in the cache
        self.n_sets =int( n_blocks / associativity )
        
        #Dictionary that holds the actual cache data
        self.data = {}
        self.set = {}
        self.domain_id_tags = {} # for cyclone

        #Pointer to the next lowest level of memory
        #Main memory gets the default None value
        self.next_level = next_level

        #Figure out spans to cut the binary addresses into block_offset, index, and tag
        self.block_offset_size = int(math.log(self.block_size, 2))
        self.index_size = int(math.log(self.n_sets, 2))

        #Initialize the data dictionary
        if next_level:
            for i in range(self.n_sets):
                index = str(bin(i))[2:].zfill(self.index_size)
                if index == '':
                    index = '0'
                #self.data[index] = {}   #Create a dictionary of blocks for each set
                self.data[index] = []    # use array instead
                self.domain_id_tags[index] = [] # for cyclone
                for j in range(associativity):
                    # isntantiate with empty tags
                    self.data[index].append((INVALID_TAG, block.Block(self.block_size, 0, False, 'x')))
                    self.domain_id_tags[index].append(('X','X')) # for cyclone

                self.set_rep_policy[index] = self.rep_policy(associativity, block_size) 

    def vprint(self, *args):
        if self.verbose == 1:
            print( " "+" ".join(map(str,args))+" ")

    # flush the cache line that contains the address from all cache hierachy
    def cflush(self, address, current_step):
        # cyclone
        cyclic_set_index = -1
        cyclic_way_index = -1
        #r = None
        r = response.Response({self.name:True}, self.cflush_time) #flush regardless 
        #Parse our address to look through this cache
        block_offset, index, tag = self.parse_address(address)
        #print(block_offset)
        #print(index)
        #print(tag)

        #Get the tags in this set
        #in_cache = list(self.data[index].keys())
        in_cache = []
        for i in range( 0, len(self.data[index]) ):
            if self.data[index][i][0] != INVALID_TAG:#'x':
                in_cache.append(self.data[index][i][0])

        #print(index)
        #print(self.data[index])
        #print(self.data[index].keys())
        #print(tag)
        
        #If this tag exists in the set, this is a hit
        if tag in in_cache:
            #if len(tag) == 0:
            #    print('false')
            
            #Delete the old block and write the new one
            #del self.data[index][tag] 
            #self.data[index][tag].write(current_step)
            for i in range( 0, len(self.data[index])):
                if self.data[index][i][0] == tag: 
                    self.data[index][i] = (INVALID_TAG, block.Block(self.block_size, current_step, False, address))
                    break
            self.set_rep_policy[index].invalidate(tag)

        # clflush from the next level of memory
        if self.next_level != None and self.next_level.name != "mem":
            self.next_level.cflush(address, current_step)

        return r, cyclic_set_index, cyclic_way_index

    # read with prefetcher
    def read(self, address, current_step, pl_opt= -1, domain_id = 'X'):
        if self.prefetcher == "none":
            
            #assert(False)
            return self.read_no_prefetch(address, current_step, pl_opt, domain_id)
        elif self.prefetcher == "nextline":
            ret = self.read_no_prefetch(hex(int(address, 16) + 1)[2:], current_step, pl_opt, domain_id)
            # prefetch the next line
            # print("nextline prefetech "+ hex(int(address, 16) + 1)[2:] )
            self.read_no_prefetch(address, current_step, pl_opt, domain_id)
            return ret 
        elif self.prefetcher == "stream":
            ret = self.read_no_prefetch(address, current_step, pl_opt, domain_id)

            # {"first": -1, "second": -1}
            # first search if it is next expected
            found= False
            for i in range(len(self.prefetcher_table)):
                entry=self.prefetcher_table[i]
                if int(address, 16) + entry["first"] == entry["second"] * 2 and entry["first"] != -1 and entry["second"] != -1 :
                    found = True
                    pref_addr = entry["second"] + int(address, 16) - entry["first"]
                    # do prefetch
                    if pref_addr >= 0:
                        # print("stream prefetech "+ hex(pref_addr)[2:])
                        self.read_no_prefetch(hex(pref_addr)[2:], current_step, pl_opt, domain_id) 
                        # update the table
                        self.prefetcher_table[i] = {"first": entry["second"], "second":pref_addr}


                elif int(address, 16) == entry["first"] + 1 or int(address, 16) == entry["first"] -1 and entry["first"] != -1:
                    # second search if it is second access
                    self.prefetcher_table[i]["second"] = int(address, 16)
                    found = True
                elif entry["first"] == int(address, 16):
                    found = True
                # then search if it is in first access

            # random evict a entry 
            if found == False:
                i = random.randint(0, self.num_prefetcher_entry-1)
                self.prefetcher_table[i] = {"first": int(address, 16), "second":-1}
            return ret
        else:
            # prefetchter not known
            assert(False)

    # pl_opt: indicates the PL cache option
    # pl_opt = -1: normal read
    # pl_opt = PL_LOCK: lock the cache line
    # pl_opt = PL_UNLOCK: unlock the cache line
    def read_no_prefetch(self, address, current_step, pl_opt= -1, domain_id = 'X'):
        # cyclone
        cyclic_set_index = -1
        cyclic_way_index = -1
        #print('pl_opt ' + str(pl_opt))
        r = None
        #Check if this is main memory
        #Main memory is always a hit
        if not self.next_level:
            r = response.Response({self.name:True}, self.hit_time)
        else:
            #Parse our address to look through this cache
            block_offset, index, tag = self.parse_address(address)
            #print(block_offset)
            #print(index)
            #print(tag)

            #Get the tags in this set
            #in_cache = list(self.data[index].keys())
            in_cache = []
            for i in range( 0, len(self.data[index]) ):
                if self.data[index][i][0] != INVALID_TAG:#'x':
                    in_cache.append(self.data[index][i][0])
            #print(index)
            #print(self.data[index])
            #print(self.data[index].keys())
            #print(tag)

            #If this tag exists in the set, this is a hit
            if tag in in_cache:
                #if len(tag) == 0:
                #    print('false')
                for i in range( 0, len(self.data[index])):
                    if self.data[index][i][0] == tag: 
                        self.data[index][i][1].read(current_step)
                        if domain_id != 'X':
                            if domain_id == self.domain_id_tags[index][i][1] and self.domain_id_tags[index][i][1] != self.domain_id_tags[index][i][0]:
                                cyclic_set_index = int(index,2)  
                                cyclic_way_index = i
                            self.domain_id_tags[index][i] = (domain_id, self.domain_id_tags[index][i][0])
                        break
                #self.data[index][tag].read(current_step)
                self.set_rep_policy[index].touch(tag, current_step)
                
                # pl cache
                if pl_opt != -1: 
                    self.set_rep_policy[index].setlock(tag, pl_opt)
                r = response.Response({self.name:True}, self.hit_time)
            else:
                #Read from the next level of memory
                r, cyclic_set_index, cyclic_way_index = self.next_level.read(address, current_step, pl_opt)
                r.deepen(self.write_time, self.name)

                #If there's space in this set, add this block to it
                if len(in_cache) < self.associativity:
                    for i in range( 0, len(self.data[index])):
                        if self.data[index][i][0] == INVALID_TAG:#'x':
                            if domain_id != 'X':
                                if domain_id == self.domain_id_tags[index][i][1] and self.domain_id_tags[index][i][1] != self.domain_id_tags[index][i][0]:
                                    cyclic_set_index = int(index, 2)  
                                    cyclic_way_index = i  
                                self.domain_id_tags[index][i] = (domain_id, self.domain_id_tags[index][i][0])
                            self.data[index][i] = (tag, block.Block(self.block_size, current_step, False, address))
                            break
                            #self.data[index][tag] = block.Block(self.block_size, current_step, False, address)
                
                    self.set_rep_policy[index].instantiate_entry(tag, current_step)
                    
                    ###if inst_victim_tag != INVALID_TAG: #instantiated entry sometimes does not replace an empty tag
                    ####we have to evict it from the cache in this scenario
                    ###    del self.data[index][inst_victim_tag]
                        
                    if pl_opt != -1:
                        self.set_rep_policy[index].setlock(tag, pl_opt)
                else:
                    #Find the victim block and replace it
                    victim_tag = self.set_rep_policy[index].find_victim(current_step)
                    #print(victim_tag)
                    # pl cache may find the victim that is partition locked
                    if victim_tag != INVALID_TAG: 
                        #oldest_tag = in_cache[0] 
                        #for b in in_cache:
                        #    if self.data[index][b].last_accessed < self.data[index][oldest_tag].last_accessed:
                        #        oldest_tag = b
                    
                        # Write the block back down if it's dirty and we're using write back
                        if self.write_back:
                            #print( self.set_rep_policy[index].candidate_tags  )
                            #print( self.set_rep_policy[index].plrutree )
                            for i in range( 0, len(self.data[index])):
                                if self.data[index][i][0] == victim_tag:
                                    if self.data[index][i][1].is_dirty():  
                            #if self.data[index][victim_tag].is_dirty():
                                        self.logger.info('\tWriting back block ' + address + ' to ' + self.next_level.name)
                                        temp = self.next_level.write(self.data[index][i][1].address, True, current_step)
                                        r.time += temp.time
                                        break
                        # Delete the old block and write the new one
                        for i in range( 0, len(self.data[index])):
                            if self.data[index][i][0] == victim_tag:
                                #del self.data[index][i][1]
                                if domain_id != 'X':
                                    if domain_id == self.domain_id_tags[index][i][1] and self.domain_id_tags[index][i][1] != self.domain_id_tags[index][i][0]:
                                        cyclic_set_index = int(index, 2)  
                                        cyclic_way_index = i
                                    self.domain_id_tags[index][i] = (domain_id, self.domain_id_tags[index][i][0])
                                self.data[index][i] = (tag, block.Block(self.block_size, current_step, False, address))
                                break
                        #del self.data[index][victim_tag]
                        self.set_rep_policy[index].invalidate(victim_tag)
                        #self.data[index][tag] = block.Block(self.block_size, current_step, False, address)
                        self.set_rep_policy[index].instantiate_entry(tag, current_step)
                        if pl_opt != -1:
                            self.set_rep_policy[index].setlock(tag, pl_opt)
        return r, cyclic_set_index, cyclic_way_index

    # pl_opt: indicates the PL cache option
    # pl_opt = -1: normal read
    # pl_opt = 1: lock the cache line
    # pl_opt = 2: unlock the cache line
    def write(self, address, from_cpu, current_step, pl_opt = -1, domain_id = 'X'):
        # cyclcone
        cyclic_set_index = -1
        cyclic_way_index = -1
        #wat is cache pls
        r = None
        if not self.next_level:
            r = response.Response({self.name:True}, self.write_time)
        else:
            block_offset, index, tag = self.parse_address(address)
            
            #in_cache = list(self.data[index].keys())
            in_cache = []
            for i in range( 0, len(self.data[index]) ):
                if self.data[index][i][0] != INVALID_TAG:#'x':
                    in_cache.append(self.data[index][i][0])

            if tag in in_cache:
                #Set dirty bit to true if this block was in cache

                #self.data[index][tag].write(current_step)
                for i in range( 0, len(self.data[index])):
                    if self.data[index][i][0] == tag:
                        if domain_id != 'X':
                            if domain_id == self.domain_id_tags[index][i][1] and self.domain_id_tags[index][i][1] != self.domain_id_tags[index][i][0]:
                                cyclic_set_index = int(index, 2)  
                                cyclic_way_index = i
                            self.domain_id_tags[index][i] = (domain_id, self.domain_id_tags[index][i][0])
                        self.data[index][i][1].write(current_step)
                        break

                self.set_rep_policy[index].touch(tag, current_step) # touch in the replacement policy
                
                if pl_opt != -1:
                    self.set_rep_policy[index].setlock(tag, pl_opt)

                if self.write_back:
                    r = response.Response({self.name:True}, self.write_time)
                else:
                    #Send to next level cache and deepen results if we have write through
                    self.logger.info('\tWriting through block ' + address + ' to ' + self.next_level.name)
                    r = self.next_level.write(address, from_cpu, current_step)
                    r.deepen(self.write_time, self.name)
            
            elif len(in_cache) < self.associativity:
                #If there is space in this set, create a new block and set its dirty bit to true if this write is coming from the CPU
                #self.data[index][tag] = block.Block(self.block_size, current_step, from_cpu, address)
                for i in range( 0, len(self.data[index])):
                    if self.data[index][i][0] == INVALID_TAG:#'x':
                        if domain_id != 'X':
                            if domain_id == self.domain_id_tags[index][i][1] and self.domain_id_tags[index][i][1] != self.domain_id_tags[index][i][0]:
                                cyclic_set_index = int(index,2)  
                                cyclic_way_index = i
                            self.domain_id_tags[index][i] = (domain_id, self.domain_id_tags[index][i][0])
                        self.data[index][i] = (tag, block.Block(self.block_size, current_step, False, address))
                        break
                
                self.set_rep_policy[index].instantiate_entry(tag, current_step)
                if self.write_back:
                    r = response.Response({self.name:False}, self.write_time)
                else:
                    self.logger.info('\tWriting through block ' + address + ' to ' + self.next_level.name)
                    r = self.next_level.write(address, from_cpu, current_step)
                    r.deepen(self.write_time, self.name)
                    if pl_opt != -1:
                        self.set_rep_policy[index].setlock(tag, pl_opt)
            
            elif len(in_cache) == self.associativity:
                
                #If this set is full, find the oldest block, write it back if it's dirty, and replace it
                victim_tag = self.set_rep_policy[index].find_victim(timestamp) 
                    
                # pl cache may find the victim that is partition locked
                # the Pl cache condition for write is not tested
                if victim_tag != INVALID_TAG: 
                    if self.write_back:
                        for i in range( 0, len(self.data[index])):
                            if self.data[index][i][0] == victim_tag:
                                if self.data[index][i][1].is_dirty():  
                        #if self.data[index][victim_tag].is_dirty():
                                    self.logger.info('\tWriting back block ' + address + ' to ' + self.next_level.name)
                                    r = self.next_level.write(self.data[index][i][1].address, from_cpu, current_step)
                                    r.deepen(self.write_time, self.name)
                                    break
                    else:
                        self.logger.info('\tWriting through block ' + address + ' to ' + self.next_level.name)
                        r, cyclic_set_index, cyclic_way_index = self.next_level.write(address, from_cpu, current_step)
                        r.deepen(self.write_time, self.name)

                    for i in range( 0, len(self.data[index])):
                        if self.data[index][i][0] == victim_tag:
                            #del self.data[index][i][1]
                            if domain_id != 'X':
                                if domain_id == self.domain_id_tags[index][i][1] and self.domain_id_tags[index][i][1] != self.domain_id_tags[index][i][0]:
                                    cyclic_set_index = int(index,2)  
                                    cyclic_way_index = i
                                self.domain_id_tags[index][i] = (domain_id, self.domain_id_tags[index][i][0])
                            self.data[index][i] = (tag, block.Block(self.block_size, current_step, False, address))
                            break
                    #del self.data[index][victim_tag]
                    self.set_rep_policy[index].invalidate(victim_tag)
                    #self.data[index][tag] = block.Block(self.block_size, current_step, from_cpu, address)
                    self.set_rep_policy[index].instantiate_entry(tag, current_step)
                    # pl cache
                    if pl_opt != -1:
                        self.set_rep_policy[index].setlock(tag, pl_opt)

                if not r:
                    r = response.Response({self.name:False}, self.write_time)

        return r, cyclic_set_index, cyclic_way_index

    def parse_address(self, address):
        #Calculate our address length and convert the address to binary string
        address_size = len(address) * 4
        binary_address = bin(int(address, 16))[2:].zfill(address_size)

        if self.block_offset_size > 0:
            block_offset = binary_address[-self.block_offset_size:]
            index = binary_address[-(self.block_offset_size+self.index_size):-self.block_offset_size]
            if index == '':
                index = '0'
            tag = binary_address[:-(self.block_offset_size+self.index_size)]
        else:
            block_offset = '0'
            if self.index_size != 0:
                index = binary_address[-(self.index_size):]
                tag = binary_address[:-self.index_size] 
            else:
                index = '0'
                tag = binary_address

        return (block_offset, index, tag)

class InvalidOpError(Exception):
    pass
