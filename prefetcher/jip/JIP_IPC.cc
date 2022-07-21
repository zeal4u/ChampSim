/***************************************************************************
For the First Instruction Prefetching Championship - IPC1

Submission ID: 91
Run-Jump-Run: Bouquet of Instruction Pointer Jumpers for High Performance Instruction Prefetching
Authors: Vishal Gupta, Neelu Shivprakash Kalani and Biswabandan Panda

***************************************************************************/

#include "ooo_cpu.h"

#include<set>
#include<stdlib.h>
#include<unordered_map>
#include<unordered_set>
#include<bitset>
#include<map>


#define L1I_PQ_SIZE 32
/***************************************************************************/
//                      PREFETCHER PARAMETERS
/***************************************************************************/


#define NRU 1

#define NUM_OF_SETS_MJT1 1024				// MJT1 = MULTIPLE TARGETS JUMP TABLE - I
#define NUM_OF_INDEX_BITS_MJT1 10
#define NUM_OF_TAG_BITS_MJT1 (25 - NUM_OF_INDEX_BITS_MJT1)
#define NUM_TARGETS_MJT1 3
#define ARRAY_OF_TARGET_LENGTH_MJT1 8
#define HISTORY_TO_MATCH_MJT1 4

#define NUM_OF_SETS_MJT2 512				// MJT2 = MULTIPLE TARGETS JUMP TABLE - II
#define NUM_OF_INDEX_BITS_MJT2 9
#define NUM_OF_TAG_BITS_MJT2 (25 - NUM_OF_INDEX_BITS_MJT2)
#define NUM_TARGETS_MJT2 8
#define ARRAY_OF_TARGET_LENGTH_MJT2 16
#define HISTORY_TO_MATCH_MJT2 4

#define PREFETCH_DEPTH 260
#define PREFETCH_DEGREE 7

#define NUM_OF_SJT_ENTRIES 7800				// SJT = SINGLE TARGET JUMP TABLE 

#define MAPPER_TABLE_SIZE 512							
#define RECENT_PREFETCH_QUEUE_SIZE 64

#define TEMPORAL_TABLE_SIZE 7150
#define RECENT_ACCESS_QUEUE_SIZE 25

#define MAX_TARGET_HIT_COUNT 3				// MJT TARGET CONFIDENCE COUNTER

#define NUM_CYCLE_OPERATE 3

#define MAX_UTILTIY_COUNTER 512				// LOOK-AHEAD PATH CONFIDENCE COUNTER
#define UTILITY_QUEUE_SIZE 32

/***************************************************************************/
/*                      HARDWARE STORAGE OVERHEAD

Hardware Tables:

Single Target Jump Table:			357500 bits	
Multiple Target Jump Table - I:			114688 bits
Multiple Target Jump Table - II:		143360 bits
Temporal Table:					357500 bits
Mapper Table:					 29184 bits

Queues:

Recent Prefetch Queue:				  1216 bits
Recent Access Queue:				   625 bits
Lookahead Prefetch Request Queue:		  2432 bits

Counters and Registers:

Lookahead prefetch confidence-counter:		     9 bits
Degree/Depth Counter:				    12 bits
Last-prefetch-IP:				    25 bits
Last-mapper-table-IP:				    25 bits
Last-prefetch-cycle:				    64 bits
L1I accesses:					     8 bits
Remaining-look-ahead-cycle:                          3 bits

Total Size:				       1046958 bits
						   127.80KB
        
*/
/***************************************************************************/


/***************************************************************************/
/*                      IP MAPPER_TABLE/
Storage: 
	Actual Tag:		48 bits
	Compressed Tag:		9 bits
	Each Entry Size:	57 bits
	#Entries:		512
	Total Size:		29184 bits
	
*/
/***************************************************************************/

//Compress the upper 6 bytes (48 bits) of a 64 bits addr to 9 bits. Compressed addr length = 9+16 = 25 bits.
class MAPPER_TABLE
{
        public:

        map<uint64_t, uint64_t> tag_array; //Actual Tag -> Compressed Tag
        map<uint64_t, uint64_t> reverse_tag_array; //Compressed Tag -> Actual Tag : Stores the same content as tag_array. Implemented separately to decrease the simulation time.
        uint64_t tag_array_ptr = 0;

	uint8_t num_lsb = 16;

        uint64_t compress_addr(uint64_t addr)
        {
		if(addr == 0)
			return addr;

		uint64_t addr_backup = addr;
	
		//We extract 16 LSBs from addr and store into lsb.
                uint64_t lsb = addr & ((1L << num_lsb) - 1);
                addr >>= num_lsb;

                uint64_t c_addr = 0;	//compressed addr

		/*Checking if we already have a mapping of 9 bits for the upper 48 bits of the addr */

                if(tag_array.find(addr) == tag_array.end())
                {
			/* We haven't compressed the upper 48 bits of the addr previously and it requires a new mapping. */

			tag_array_ptr++;

			/*If the number of mappings exceed the Mapper Table's size, we use FIFO replacement, hence setting the array pointer to index 0. */

			if(tag_array_ptr == MAPPER_TABLE_SIZE)
				tag_array_ptr = 0;

	
			if(reverse_tag_array.find(tag_array_ptr) == reverse_tag_array.end())
			{
				//Not replacing an older mapping

				tag_array[addr] = tag_array_ptr;
				reverse_tag_array[tag_array_ptr] = addr;
			}
			else
			{
				//Replacing an older mapping

				for(auto it: tag_array)
					if(it.second == tag_array_ptr)
					{
						tag_array.erase(it.first);
						break;
					}

				tag_array[addr] = tag_array_ptr;
                                reverse_tag_array[tag_array_ptr] = addr;

			}

                        c_addr = tag_array_ptr;
                }
                else
                {
			/* If the upper 48 bits of the addr already have a corresponding 9 bits value mapped, then use the 9 bit value from the table. */

                        c_addr = tag_array[addr];
                }

		//Append the 16 lsb to the compressed 9 bits and return the compressed addr.
                c_addr <<= num_lsb;
                c_addr |= lsb;

                return c_addr;
        }

        uint64_t uncompress_addr(uint64_t addr)
        {
		/*Finds the 48 bits value mapped with the 9 bits value from the reverse mapping. */

		if(addr == 0)
			return addr;

		uint64_t addr_backup = addr;

                uint64_t lsb = addr & ((1L << num_lsb) - 1);
                addr >>= num_lsb;

                uint64_t uc_addr = 0;

                uc_addr = reverse_tag_array[addr];
                uc_addr <<= num_lsb;
                uc_addr |= lsb;

		return uc_addr;
        }
};

MAPPER_TABLE mapper_table;						//MAPPER_TABLE TABLE

/***************************************************************************/
/*                      LOOK-AHEAD PATH SELECTOR
Storage: 
        Look-ahead from Temporal Table Target IP:	64 * 19 bits
	Look-ahead from Last Prefetch IP:		64 * 19 bits
        Look-ahead Path Confidence Counter:		9 bits
        Total Size:					2441 bits 
*/
/***************************************************************************/

/* NOTE: The LOOKAHEAD_PATH_SELECTOR Class has two queues (temporal_table_pref & normal_path_pref) to store recent prefetch requests - one for the prefetch requests that we make by performing look-ahead from the Temporal Table Target IP and another for the prefetch requests that we make by performing look-ahead from the Last Prefetch IP. Do not confuse these queues with the RECENT PREFETCH QUEUE (declared later on) which we use to filter prefetch requests. */

class LOOKAHEAD_PATH_SELECTOR
{
	public:
	vector<uint64_t> temporal_table_pref;			//Look-ahead from Temporal Table Target IP
	vector<uint64_t> normal_path_pref;			//Look-ahead from Last Prefetch IP

	uint64_t utility_counter = MAX_UTILTIY_COUNTER / 2;	//Look-ahead Path Confidence Counter
	
	void insert(uint64_t addr, bool is_temporal_table)
        {
                addr >>= 6; // Removing last 6 bits to make it cache block aligned

		//Inserting in different queues by checking is_temporal_table flag.
		//Checking for duplicates and erasing them before inserting.

		if(is_temporal_table)
		{
			for(auto it = temporal_table_pref.begin(); it != temporal_table_pref.end(); it++)
				if((*it) == addr)
				{
					it = temporal_table_pref.erase(it);
					if(it == temporal_table_pref.end())
						break;
				}
			if(temporal_table_pref.size() == UTILITY_QUEUE_SIZE)
				temporal_table_pref.erase(temporal_table_pref.begin());
			temporal_table_pref.push_back(addr);
		}
		else
		{
			for(auto it = normal_path_pref.begin(); it != normal_path_pref.end(); it++)
				if((*it) == addr)
				{
					it = normal_path_pref.erase(it);
					if(it == normal_path_pref.end())
						break;
				}
			if(normal_path_pref.size() == UTILITY_QUEUE_SIZE)
				normal_path_pref.erase(normal_path_pref.begin());
			normal_path_pref.push_back(addr);

		}

        }

        void mark_hit(uint64_t addr)
        {
                addr >>= 6;

		/*Checking if the addr is present in the recent prefetch queue of the particular path and if it is, modifying the confidence counter, incrementing by 2 to favour the temporal table target path and decrementing by 1 to favour the last prefetch IP path. Note that we favour the temporal table target path with a higher weightage as it shows higher utility. */

                for(int i = 0; i < temporal_table_pref.size(); i++)
                        if(temporal_table_pref[i] == addr)
			{
				if(utility_counter + 2 < MAX_UTILTIY_COUNTER)
					utility_counter += 2;
				break;
			}
	
		for(int i = 0; i < normal_path_pref.size(); i++)
                        if(normal_path_pref[i] == addr)
			{
				if(utility_counter > 0)
					utility_counter -= 1;
				break;
			}

        }
	
	bool is_temporal_table_path_good()
	{
		return utility_counter >= (MAX_UTILTIY_COUNTER / 2);
	}

};

LOOKAHEAD_PATH_SELECTOR lookahead_path_selector;				//LOOK_AHEAD PATH SELECTOR

/***************************************************************************/
/*                      RECENT PREFETCH QUEUE
Storage: 
        Queue:		64 * 19 bits
        Total Size:     1216 bits 
*/
/***************************************************************************/


class RECENT_PREFETCH_QUEUE
{
	public:
	vector<uint64_t> queue;

	void insert(uint64_t addr)
	{
		addr >>= 6; // Remove last 6 bits to make it cache block aligned

		//Checking for duplicates and erasing them before inserting.

		for(auto it = queue.begin(); it != queue.end(); it++)
			if(*it == addr)
			{
				it = queue.erase(it);
				if(it == queue.end())
					break;
			}
		

		if(queue.size() == RECENT_PREFETCH_QUEUE_SIZE)
			queue.erase(queue.begin());
		
		queue.push_back(addr);
		
	}

	bool find(uint64_t addr)
	{
		addr >>= 6;

		for(int i = 0; i < queue.size(); i++)
			if(queue[i] == addr)
				return true;
		return false;
	}

	bool issue_prefetch(O3_CPU *o3_cpu, uint64_t addr, int lookahead_path) //lookahead_path == 0 (cache_operate); 1 (cycle_operate); 2(cycle_operate_other/temporal_table_path)
        {
		
		/* If address is not present in Recent Prefetch Queue, then we check if the prefetch request lies on the lookahead path we follow in cycle_operate. If yes, then we make the prefetch request only if the path's confidence is high. If no, then it means that the prefetch request is from cache_operate so we make the prefetch request. */
		
                if(!find(addr))
                {
			if(lookahead_path == 2)
			{
				lookahead_path_selector.insert(addr,true);
				if(!lookahead_path_selector.is_temporal_table_path_good())
					return true; //Don't prefetch this path
			}
			else if(lookahead_path == 1)
			{
				lookahead_path_selector.insert(addr,false);
				if(lookahead_path_selector.is_temporal_table_path_good())
					return true; //Don't Prefetch this path
			}
			
                        insert(addr);
                        addr = mapper_table.uncompress_addr(addr);
                        o3_cpu->prefetch_code_line(addr);
                        return true;
                }
                return false;
        }
};

RECENT_PREFETCH_QUEUE recent_prefetch_queue;

class CACHE_ENTRY
{
	public:
	uint64_t target; // 25 bits
	uint64_t nru; // 1 bit

	CACHE_ENTRY():target(0), nru(0)
	{}

	CACHE_ENTRY(uint64_t target): target(target), nru(0)
	{}
};

class FULLY_ASSOCIATIVE_CACHE {
        public:
	unordered_map<uint64_t, CACHE_ENTRY> cache_entries; //Key (Trigger IP): 25 bits
	int NUM_CACHE_ENTRIES;
	bool is_temporal_table;

	FULLY_ASSOCIATIVE_CACHE(int size, bool is_temporal_table)
	{
		NUM_CACHE_ENTRIES = size;
		this->is_temporal_table = is_temporal_table;
	}

        void insert(uint64_t ip, uint64_t target)
        {
	
		int done = 0;

		/* First we iterate through the map for finding max_nru. If we do not find an entry with max_nru, we check sjt_occupancy and add a new entry if it's not equal to the maximum SJT size. If it is, then we increment nru for all entries and repeat to find the max_nru. */ 

		while(!done)
		{	
			auto it = cache_entries.begin();
			
			if(cache_entries.size() < NUM_CACHE_ENTRIES && !done)
			{
				//inserting new map entry if map size is not greater than maximum SJT size.	

				cache_entries.insert({ip, CACHE_ENTRY(target)});
				done = 1;
				break;
			}

			if(!done && is_temporal_table)
			{
				it = cache_entries.begin();

				delete_entry(it->first);
				cache_entries.insert({ip, CACHE_ENTRY(target)});
				done = 1;
				break;
			}

			if(!done)
			{
				it = cache_entries.begin();

				while(it != cache_entries.end())
				{
					if(it->second.nru == NRU)
					{
						delete_entry(it->first);
						cache_entries.insert({ip, CACHE_ENTRY(target)});
						done = 1;
						break;
					}
					it++;
				}
			}

			if(!done)
			{
				//Incrementing nru for all map entries.

				it = cache_entries.begin();

				while(it != cache_entries.end())
				{
					if(it->second.nru < NRU)
						it->second.nru++;
		
					it++;
				}
			}
		}
	}

	bool find(uint64_t ip)
	{
		if(cache_entries.find(ip) == cache_entries.end())
			return false;
		
		return true;	
	}

	void insert_or_update(uint64_t ip, uint64_t target)
	{
		auto it = cache_entries.find(ip);
		if(it == cache_entries.end())
			insert(ip, target);			//Insert
		else
		{
			it->second.target = target;		//Update
			it->second.nru = 0;
		}
	}

	void delete_entry(uint64_t ip)
	{
		if(cache_entries.find(ip) == cache_entries.end())
			return;
	
		cache_entries.erase(ip);
	}

	void update_nru_on_hit(uint64_t ip)
	{
		if(cache_entries.find(ip) == cache_entries.end())
                        return;

		cache_entries[ip].nru = 0;
	}

	uint64_t get_target(uint64_t ip)
	{
		if(cache_entries.find(ip) == cache_entries.end())
                        return 0;

		return cache_entries[ip].target; 
	}

};

/***************************************************************************/
/*                      SINGLE TARGET JUMP TABLE
Storage:
	Trigger IP:	25 bits	
	Target IP:	25 bits
	NRU:		1 bit
	Entry Size:	51 bits
	#Entries:	7800
	Total Size:	397800 bits
*/
/***************************************************************************/


FULLY_ASSOCIATIVE_CACHE sjt(NUM_OF_SJT_ENTRIES, false);

/***************************************************************************/
/*                      TEMPORAL TABLE
Storage:
        Trigger IP:     25 bits 
        Target IP:      25 bits
        Entry Size:     50 bits
        #Entries:       7150
        Total Size:     357500 bits
*/
/***************************************************************************/

FULLY_ASSOCIATIVE_CACHE temporal_table(TEMPORAL_TABLE_SIZE, true);


class MULTIPLE_JUMP_TABLE_ENTRY {
        public:
        uint64_t tag;           
        vector<uint64_t> target;        
	vector<uint64_t> target_hit_count;	//Confidence Counters Per Target
	vector<uint8_t> history;		//Array of Targets
	int ARRAY_OF_TARGET_LENGTH;

	int HISTORY_TO_MATCH;			/*The number of targets in the temporal sequence we match with the array of targets */

        MULTIPLE_JUMP_TABLE_ENTRY(int num_targets, int array_of_target_length, int history_to_match) {
                tag = 0;
		
		target.resize(num_targets);
		target_hit_count.resize(num_targets);		
                ARRAY_OF_TARGET_LENGTH = array_of_target_length;
		HISTORY_TO_MATCH = history_to_match;
        }

	void add_history_index(uint8_t index)
	{
		if(index >= target.size())
			return;
	
		target_hit_count[index]++;
		
		/* If confidence counter of the target saturates, we decrement the confidence counters of all the targets by one. */

		if(target_hit_count[index] == MAX_TARGET_HIT_COUNT)
		{
			for(int i = 0; i < target_hit_count.size(); i++)
				if(target_hit_count[i] > 0)
					target_hit_count[i]-=1;
		}	


		history.push_back(index);
		if(history.size() > ARRAY_OF_TARGET_LENGTH)
			history.erase(history.begin());
	}

	uint8_t max_target_hit_index()
	{
		//Returns the target with the highest confidence counter.

		int max_index = 0;
		for(int i = 1; i < target_hit_count.size(); i++)
			if(target_hit_count[max_index] < target_hit_count[i])
				max_index = i;

		return max_index;
	}

	uint8_t get_unused_index()
        {
		/* Returns the least recently used target by referring the array of targets (history) which stores the temporal sequence in which the targets were accessed. */

                bool vals[target.size()] = {false};
		vector<uint8_t> lru;
                for(int i = history.size() - 1; i >= 0; i--)
		{
			if(vals[history[i]] == false)
			{
				vals[history[i]] = true;
				lru.push_back(history[i]);
			}
	
		}
                for(int i = 0; i < target.size(); i ++)
                        if(!vals[i])
                                return i;

		return *(lru.rbegin());
        }

	uint8_t get_history_index()
	{
		if(history.size() == 0)
			return max_target_hit_index();
		if(history.size() < HISTORY_TO_MATCH)
			return max_target_hit_index(); 

		/* We take the last HISTORY_TO_MATCH number of elements from the array of targets (history) and match the whole sequence with the remaining elements. For example, if the array of targets (history) stores 8 elements, and HISTORY_TO_MATCH is 4, then we compare the last 4 elements of the array of targets (history) (4-5-6-7) with the following elements (3-4-5-6), (2-3-4-5), (1-2-3-4), (0-1-2-3). If all the elements match with any of the above sequences, the target element which we return will be the one after the sequence ends, i.e. 7,6,5,4, respectively. If there is no match, then we return the target with the maximum confidence. */ 

		for(int i = history.size() - HISTORY_TO_MATCH - 1; i >= 0; i--)
		{
			int cnt = 0;
			for(int j = 0;j < HISTORY_TO_MATCH; j++)
				if(history[i + j] == history[(history.size() - HISTORY_TO_MATCH) + j])
					cnt++;
				else
					break;

			if(cnt == HISTORY_TO_MATCH)
			{
				if((i + HISTORY_TO_MATCH) < history.size())
				{
					return history[ i + HISTORY_TO_MATCH ];
				}
			}
		}

		//History does not match, return last element
		return max_target_hit_index();
	}

};

//MJTs are Direct-Mapped

class MULTIPLE_JUMP_TABLE {
	public:
	vector<MULTIPLE_JUMP_TABLE_ENTRY> mjt_entries;
	int NUM_SETS;
	int NUM_TARGETS;
	int ARRAY_OF_TARGET_LENGTH;
	int NUM_INDEX_BITS;
	int NUM_TAG_BITS;

	MULTIPLE_JUMP_TABLE(int num_sets, int num_targets, int array_of_target_length, int num_index, int num_tag, int history_to_match)
	{
		for(int i = 0 ; i < num_sets; i++)
		{
			mjt_entries.push_back(MULTIPLE_JUMP_TABLE_ENTRY(num_targets, array_of_target_length, history_to_match));
		}
		
		NUM_SETS = num_sets;
		NUM_TARGETS = num_targets;
		ARRAY_OF_TARGET_LENGTH = array_of_target_length;
		NUM_INDEX_BITS = num_index;
		NUM_TAG_BITS = num_tag;
	}

	int bit_shift = 2; //Bits to shift to take index bits from

        int get_index(uint64_t ip)
        {
                ip >>= bit_shift;
                return (ip & ((1L << NUM_INDEX_BITS) - 1));
        }

        int get_tag(uint64_t ip)
        {
                int lsb = (ip & ((1L << bit_shift) - 1));

                ip >>= bit_shift;
                ip >>= NUM_INDEX_BITS;

                ip <<= bit_shift;
                return (ip | lsb);
        }
        uint64_t recreate_ip(int index, int tag)
        {
                int lsb = (tag & ((1L << bit_shift) - 1));

                tag >>= bit_shift;
                tag <<= NUM_INDEX_BITS;

                tag |= index;

                tag <<= bit_shift;
                tag |= lsb;

                return tag;
        }

	void insert(int index, uint64_t hash_ip, vector<uint64_t> target, vector<uint8_t> history, vector<uint64_t> target_hit_count)
	{

		mjt_entries[index].target.clear();
		mjt_entries[index].target.resize(NUM_TARGETS);
		for(int j = 0; j < target.size(); j++)
			mjt_entries[index].target[j] = target[j];

		mjt_entries[index].target_hit_count.clear();
		mjt_entries[index].target_hit_count.resize(NUM_TARGETS);
		for(int j = 0; j < target_hit_count.size(); j++)
		mjt_entries[index].target_hit_count[j] = target_hit_count[j];

		mjt_entries[index].history.clear();
	
		for(int j = 0; j < history.size(); j++)
			mjt_entries[index].add_history_index(history[j]);

		mjt_entries[index].add_history_index(target.size());
		mjt_entries[index].tag = get_tag(hash_ip);
	}

	void add_target(int index, uint64_t target)
	{
		int i;
		
		for(i = 0; i < NUM_TARGETS; i++)
                {
                        if(mjt_entries[index].target[i] == target)
                        {
                                mjt_entries[index].add_history_index(i);
				return;
                        }
                }

		for(i = 0; i < NUM_TARGETS; i++)
		{
			if(mjt_entries[index].target[i] == 0)
			{
				mjt_entries[index].target[i] = target;
				mjt_entries[index].add_history_index(i);
				return;	
			}
		}

		if(i == NUM_TARGETS)
		{
			i = mjt_entries[index].get_unused_index();
			mjt_entries[index].target[i] = target;
			mjt_entries[index].add_history_index(i);
		}
	}

	bool find(int index, uint64_t hash_ip)
        {
                uint64_t tag = get_tag(hash_ip);
		if(mjt_entries[index].tag == tag)
			return true;
                return false;
        }

	void delete_mjt(int index)
        {
                mjt_entries[index].tag = 0;
		for(int i = 0; i < NUM_TARGETS; i++)
			mjt_entries[index].target[i] = 0;
		mjt_entries[index].history.clear();
		mjt_entries[index].target.clear();
		mjt_entries[index].target.resize(NUM_TARGETS);

		mjt_entries[index].target_hit_count.clear();
                mjt_entries[index].target_hit_count.resize(NUM_TARGETS);
        }

};

/***************************************************************************/
/*                      MULTIPLE TARGET JUMP TABLE - I
Storage:
        Tag:			16 bits 
        Target IP:		3 * 25 bits
	Array of Targets:	8 * 2 bits
	Target Confidence:	3 * 2 bits
        Entry Size:		112 bits
        #Entries:		1024
        Total Size:		114688 bits
*/
/***************************************************************************/

MULTIPLE_JUMP_TABLE multiple_jump_table1(NUM_OF_SETS_MJT1, NUM_TARGETS_MJT1, ARRAY_OF_TARGET_LENGTH_MJT1, NUM_OF_INDEX_BITS_MJT1, NUM_OF_TAG_BITS_MJT1, HISTORY_TO_MATCH_MJT1);

/***************************************************************************/
/*                      MULTIPLE TARGET JUMP TABLE - II
Storage:
	Tag:                    16 bits 
        Target IP:              8 * 25 bits
        Array of Targets:       16 * 3 bits
        Target Confidence:      8 * 2 bits
        Entry Size:             280 bits
        #Entries:               512
        Total Size:             143360 bits
*/
/***************************************************************************/

MULTIPLE_JUMP_TABLE multiple_jump_table2(NUM_OF_SETS_MJT2, NUM_TARGETS_MJT2, ARRAY_OF_TARGET_LENGTH_MJT2, NUM_OF_INDEX_BITS_MJT2, NUM_OF_TAG_BITS_MJT2, HISTORY_TO_MATCH_MJT2);

/***************************************************************************/
/*                      RECENT ACCESS QUEUE
Storage:
        Entry Size:             25 bits
        #Entries:               25
        Total Size:             625 bits
*/
/***************************************************************************/

vector<uint64_t> recent_access_queue;

uint64_t last_prefetch_cycle = 0, last_prefetch_ip = 0, last_prefetch_ip_other = 0;
int num_cycle_operate_times = NUM_CYCLE_OPERATE;

int num_accesses = 0;


void O3_CPU::prefetcher_initialize()
{
}

void O3_CPU::prefetcher_branch_operate(uint64_t ip, uint8_t branch_type, uint64_t branch_target)
{
	uint64_t ip_backup = ip;

	uint64_t hash_ip = ip;
	
	hash_ip = mapper_table.compress_addr(hash_ip);
	branch_target = mapper_table.compress_addr(branch_target);

        if(branch_target != 0)
        {
                int processed_flag = 0;
	
                int mjt1_index = multiple_jump_table1.get_index(hash_ip); 
                bool mjt1_present = multiple_jump_table1.find(mjt1_index, hash_ip);

		int mjt2_index = multiple_jump_table2.get_index(hash_ip); 
                bool mjt2_present = multiple_jump_table2.find(mjt2_index, hash_ip);
		
		bool sjt_present = sjt.find(hash_ip);

		//Case 1: If present in MJT2, add target.
                if(mjt2_present)
                {
			//add target
			multiple_jump_table2.add_target(mjt2_index, branch_target);
			processed_flag = 1;
		}

		//Case 2: Hit in MJT1, check for MJT1 to MJT2 migration.
		if(!processed_flag && mjt1_present)
                {
			int i = 0;
			for(i = 0; i < NUM_TARGETS_MJT1; i++)
				if(multiple_jump_table1.mjt_entries[mjt1_index].target[i] == 0 || multiple_jump_table1.mjt_entries[mjt1_index].target[i] == branch_target)
					break;
			
			if(i == NUM_TARGETS_MJT1) //MJT1 to MJT2 migration
			{
				vector<uint64_t> target_arr;
				for(i = 0; i < NUM_TARGETS_MJT1; i++)
					target_arr.push_back(multiple_jump_table1.mjt_entries[mjt1_index].target[i]);
				
				target_arr.push_back(branch_target);
				multiple_jump_table2.insert(mjt2_index, hash_ip, target_arr, multiple_jump_table1.mjt_entries[mjt1_index].history, multiple_jump_table1.mjt_entries[mjt1_index].target_hit_count);
	
				multiple_jump_table1.delete_mjt(mjt1_index);
			}
			else
			{
				//add target
				multiple_jump_table1.add_target(mjt1_index, branch_target);
			}
                        processed_flag = 1;
                }

		//Case 3: Hit in SJT, check for SJT to MJT1 migration
		if(!processed_flag && sjt_present)	
		{
			if(branch_target != sjt.get_target(hash_ip))
			{
				//migrating to MJT1
				vector<uint64_t> target_arr;
				target_arr.push_back(sjt.get_target(hash_ip));
				target_arr.push_back(branch_target);

				vector<uint8_t> temp_buf;
                                vector<uint64_t> temp_buf1;
                                multiple_jump_table1.insert(mjt1_index, hash_ip, target_arr, temp_buf, temp_buf1);
				
				sjt.delete_entry(hash_ip);
			}
			else
			{
				sjt.update_nru_on_hit(hash_ip);
			}
			processed_flag = 1;
		}

		//Case 4: Not present in SJT and MJT.
                if(!processed_flag)
                {
			//Inserting all branch IPs to SJT first
			sjt.insert(hash_ip, branch_target);	
			processed_flag = 1;
		}
		//Prefetching the branch target of the current IP
		recent_prefetch_queue.issue_prefetch(this, branch_target, 0);
	}
}

uint32_t O3_CPU::prefetcher_cache_operate(uint64_t addr, uint8_t cache_hit, uint8_t prefetch_hit, uint32_t metadata_in) //All addresses are virtual addresses
{	
	num_accesses++;
	
	if(num_accesses == 256)
	{
		//Lookahead Path Confidence Counter is reset to default value after every 256 L1-I accesses.
		num_accesses = 0;
		lookahead_path_selector.utility_counter = (MAX_UTILTIY_COUNTER / 2);
	}

	addr = mapper_table.compress_addr(addr);

	//Updating lookahead path confidence counter
	lookahead_path_selector.mark_hit(addr);

        int prefetch_depth= PREFETCH_DEPTH; 
        int prefetch_degree= PREFETCH_DEGREE; 
	uint64_t pref_gen_this_cycle = 0;
	uint64_t new_cycle_operate_ip = 0;

	last_prefetch_cycle = current_core_cycle[0];

	if(recent_access_queue.size() > 0 && cache_hit == 0)
        {
		//If the current access is a cache miss, adding <leader IP, follower IP> pair to temporal table.
		temporal_table.insert_or_update(*(recent_access_queue.begin()), addr);
        }
	
	if(recent_access_queue.size() == RECENT_ACCESS_QUEUE_SIZE)
		recent_access_queue.erase(recent_access_queue.begin());
	recent_access_queue.push_back(addr);

        uint64_t prev_line = addr >> (LOG2_BLOCK_SIZE);
        uint64_t cur_ip = addr;
        uint64_t pref_ip, hash_ip;
        int jt_index, jt_way, mjt1_index,mjt2_index, processed_flag;
	bool mjt1_present, mjt2_present, sjt_present, temporal_table_hit = false;
	
	if(!temporal_table_hit && temporal_table.find(cur_ip))
	{
		temporal_table_hit = true;

		//saving the temporal table target for lookahead in the subsequenct cycles.
		new_cycle_operate_ip = temporal_table.get_target(cur_ip);
	}

	/* Performing lookahead till prefetch_depth reaches maximum lookahead depth or prefetch_degree reaches maximum prefetch_degree */

	int i;
	for(i = 0; i < prefetch_depth; i++)
        {
                pref_ip = 0;
                processed_flag = 0;
                hash_ip = cur_ip;

		if(!temporal_table_hit && temporal_table.find(cur_ip))
		{
			temporal_table_hit = true;

			//saving the temporal table target for lookahead in the subsequenct cycles.
			new_cycle_operate_ip = temporal_table.get_target(cur_ip);
		}

		//Case 1: Search in MJT2
                mjt2_index = multiple_jump_table2.get_index(hash_ip);
                mjt2_present = multiple_jump_table2.find(mjt2_index, hash_ip);

                if(mjt2_present)
                {
                        uint8_t last_index = multiple_jump_table2.mjt_entries[mjt2_index].get_history_index();

                        pref_ip = multiple_jump_table2.mjt_entries[mjt2_index].target[last_index];

                        if (L1I_bus.lower_level->get_occupancy(3, 0) < (L1I_PQ_SIZE) && pref_ip != 0 && recent_prefetch_queue.issue_prefetch(this,pref_ip, 0))
                        {
				pref_gen_this_cycle++;
                        }
                        processed_flag = 1;
                }

                //Case 2: Search in MJT1
                mjt1_index = multiple_jump_table1.get_index(hash_ip);
                mjt1_present = multiple_jump_table1.find(mjt1_index, hash_ip);
		
		if(!processed_flag && mjt1_present)
                {
	
			uint8_t last_index = multiple_jump_table1.mjt_entries[mjt1_index].get_history_index();

			pref_ip = multiple_jump_table1.mjt_entries[mjt1_index].target[last_index];

			if (L1I_bus.lower_level->get_occupancy(3, 0) < (L1I_PQ_SIZE) && pref_ip != 0 && recent_prefetch_queue.issue_prefetch(this,pref_ip, 0))
			{
				pref_gen_this_cycle++;	
			}
                        processed_flag = 1;
                }

		sjt_present = sjt.find(hash_ip);

		if(!processed_flag && sjt_present)
		{
			uint64_t sjt_target = sjt.get_target(hash_ip);
			//Hit in SJT, prefetch target, mark processed. 
			if((sjt_target >> (LOG2_BLOCK_SIZE))!= prev_line)
                        {
                                if (L1I_bus.lower_level->get_occupancy(3, 0) < L1I_PQ_SIZE)
                                {
                                        pref_ip = sjt_target;
										
					if(recent_prefetch_queue.issue_prefetch(this, sjt_target, 0))
						pref_gen_this_cycle++;
				}
			}
                        processed_flag = 1;
		}

		if(!processed_flag)
                {
                        //Cannot find in any tables, do next line prefetching and mark processed.
                        if((cur_ip >> (LOG2_BLOCK_SIZE) ) != prev_line)
                        {
				if (L1I_bus.lower_level->get_occupancy(3, 0) < L1I_PQ_SIZE && recent_prefetch_queue.issue_prefetch(this, cur_ip, 0))
				{
					pref_gen_this_cycle++;
				}
                        }
                        processed_flag = 1;
                }

                //Update prev_line and cur_ip for next iteration.
                prev_line = cur_ip >> (LOG2_BLOCK_SIZE);

                if(pref_ip != 0)
                        cur_ip = pref_ip;
		else
                {
                        uint64_t msb = cur_ip >> 16;
                        cur_ip ++;

                        if((cur_ip >> 16) != msb)
                        {
                                cur_ip--;
                                uint64_t org_addr = mapper_table.uncompress_addr(cur_ip);
                                org_addr++;
                                cur_ip = mapper_table.compress_addr(org_addr);
                        }
                }

		if(pref_gen_this_cycle > prefetch_degree)
                        break;
	}

	if(!temporal_table_hit && temporal_table.find(cur_ip))
	{
		temporal_table_hit = true;

		//saving the temporal table target for lookahead in the subsequenct cycles.
		new_cycle_operate_ip = temporal_table.get_target(cur_ip);
	}



	if(temporal_table_hit)
	{
		//prefetching temporal table target in case of temporal table hit.

		uint64_t prefetch_addr = new_cycle_operate_ip;
		recent_prefetch_queue.issue_prefetch(this,prefetch_addr, 0);
		last_prefetch_ip = cur_ip; //new_cycle_operate_ip;
		last_prefetch_ip_other = new_cycle_operate_ip;
	}
	else
	{
		last_prefetch_ip = cur_ip;
		last_prefetch_ip_other = 0;
	}
	num_cycle_operate_times = NUM_CYCLE_OPERATE;
	return metadata_in;
}

void l1i_prefetcher_cycle_operate_other(O3_CPU* o3_cpu)
{

	int prefetch_depth = PREFETCH_DEPTH;
	int prefetch_degree = 1;	//prefetch degree is set to one for prefetching in cycle_operate

	uint64_t pref_gen_this_cycle = 0;
	uint64_t addr = last_prefetch_ip_other;		//performing lookahead from the temporal table target IP

	uint64_t prev_line = addr >> (LOG2_BLOCK_SIZE);
        uint64_t cur_ip = addr;
        uint64_t pref_ip, hash_ip;
        int jt_index, jt_way, mjt1_index, mjt2_index, processed_flag;

	bool sjt_present, mjt1_present, mjt2_present;

        for(int i = 0; i < prefetch_depth ; i++)
        {
                pref_ip = 0;
                processed_flag = 0;
                hash_ip = cur_ip;

		//Case 1: Search in MJT2
                mjt2_index = multiple_jump_table2.get_index(hash_ip);
                mjt2_present = multiple_jump_table2.find(mjt2_index, hash_ip);

                if(mjt2_present)
                {
                        uint8_t last_index = multiple_jump_table2.mjt_entries[mjt2_index].get_history_index();

                        pref_ip = multiple_jump_table2.mjt_entries[mjt2_index].target[last_index];

                        if (o3_cpu->L1I_bus.lower_level->get_occupancy(3, 0) < (L1I_PQ_SIZE) && pref_ip != 0 && recent_prefetch_queue.issue_prefetch(o3_cpu,pref_ip, 2))
                        {
				pref_gen_this_cycle++;
                        }
                        processed_flag = 1;
                }

                //Case 2: Search in MJT
                mjt1_index = multiple_jump_table1.get_index(hash_ip);
                mjt1_present = multiple_jump_table1.find(mjt1_index, hash_ip);

                if(mjt1_present)
                {
                        //Hit in MJT, prefetch all targets, mark processed. 

			uint8_t last_index = multiple_jump_table1.mjt_entries[mjt1_index].get_history_index();

			pref_ip = multiple_jump_table1.mjt_entries[mjt1_index].target[last_index];
			
                        if (o3_cpu->L1I_bus.lower_level->get_occupancy(3, 0) < (L1I_PQ_SIZE) && pref_ip != 0 && recent_prefetch_queue.issue_prefetch(o3_cpu, pref_ip, 2))
                        {
				pref_gen_this_cycle++;
                        }
                        processed_flag = 1;
                }

		sjt_present = sjt.find(hash_ip);

                if(!processed_flag && sjt_present)
                {
                        uint64_t sjt_target = sjt.get_target(hash_ip);
                        //Hit in SJT, prefetch target, mark processed.
                        if((sjt_target >> (LOG2_BLOCK_SIZE))!= prev_line)
                        {
                                if (o3_cpu->L1I_bus.lower_level->get_occupancy(3, 0) < L1I_PQ_SIZE)
                                {
                                        pref_ip = sjt_target;
			
					if(recent_prefetch_queue.issue_prefetch(o3_cpu,pref_ip, 2))
						pref_gen_this_cycle++;

                                }
                        }
                        processed_flag = 1;
                }

                if(!processed_flag)
                {
                        //Cannot find in any tables, do next line prefetching and mark processed flag.
                        if((cur_ip >> (LOG2_BLOCK_SIZE)) != prev_line)
                        {
				if (o3_cpu->L1I_bus.lower_level->get_occupancy(3, 0) < L1I_PQ_SIZE && recent_prefetch_queue.issue_prefetch(o3_cpu, cur_ip, 2)) 
				pref_gen_this_cycle++;
                        }
                        processed_flag = 1;
                }

                //Update prev_line and cur_ip for next iteration.
                prev_line = cur_ip >> (LOG2_BLOCK_SIZE);

                if(pref_ip != 0)
                        cur_ip = pref_ip;
                else
		{
			uint64_t msb = cur_ip >> 16;                
			cur_ip ++;

			if((cur_ip >> 16) != msb)
			{
				cur_ip--;
				uint64_t org_addr = mapper_table.uncompress_addr(cur_ip);
				org_addr++;
				cur_ip = mapper_table.compress_addr(org_addr);
			}
		}

		if(pref_gen_this_cycle > prefetch_degree)
                        break;

	}

        last_prefetch_ip_other= cur_ip;
}


void O3_CPU::prefetcher_cycle_operate()
{

	if(num_cycle_operate_times <= 0)
		return;
	
	num_cycle_operate_times--;


	int prefetch_depth = PREFETCH_DEPTH;
	int prefetch_degree = 1;	//prefetch degree is set to one for prefetching in cycle_operate

// If the gap between cache_operate and cycle_operate is less than 2, then return
/* We do this so that if the processor is continuously sending requests to the L1-I, we don't send prefetch requests that might get in the way of those demand requests. So if there are no processor requests for 2 cycles, we start the lookahead process in cycle_operate. */
	if((current_core_cycle[0] - last_prefetch_cycle) < 2) 
		return;

	uint64_t pref_gen_this_cycle = 0;
	uint64_t addr = last_prefetch_ip;	//Performing lookahead from the last prefetch IP

	if(addr == 0)
		return;

	//performing lookahead from the temporal table target IP	
	if(last_prefetch_ip_other)
		l1i_prefetcher_cycle_operate_other(this);

	uint64_t prev_line = addr >> (LOG2_BLOCK_SIZE);
        uint64_t cur_ip = addr;
        uint64_t pref_ip, hash_ip;
        int jt_index, jt_way, mjt1_index, mjt2_index, processed_flag;

	bool sjt_present, mjt1_present, mjt2_present;

        for(int i = 0; i < prefetch_depth ; i++)
        {
                pref_ip = 0;
                processed_flag = 0;
                hash_ip = cur_ip;

		//Case 1: Search in MJT2
                mjt2_index = multiple_jump_table2.get_index(hash_ip);
                mjt2_present = multiple_jump_table2.find(mjt2_index, hash_ip);

                if(mjt2_present)
                {
                        uint8_t last_index = multiple_jump_table2.mjt_entries[mjt2_index].get_history_index();

                        pref_ip = multiple_jump_table2.mjt_entries[mjt2_index].target[last_index];

                        if (L1I_bus.lower_level->get_occupancy(3, 0) < (L1I_PQ_SIZE) && pref_ip != 0 && recent_prefetch_queue.issue_prefetch(this,pref_ip, 1))
                        {
				pref_gen_this_cycle++;
                        }
                        processed_flag = 1;
                }

                //Case 2: Search in MJT
                mjt1_index = multiple_jump_table1.get_index(hash_ip);
                mjt1_present = multiple_jump_table1.find(mjt1_index, hash_ip);

                if(mjt1_present)
                {
			uint8_t last_index = multiple_jump_table1.mjt_entries[mjt1_index].get_history_index();

			pref_ip = multiple_jump_table1.mjt_entries[mjt1_index].target[last_index];
			
                        if (L1I_bus.lower_level->get_occupancy(3, 0) < (L1I_PQ_SIZE) && pref_ip != 0 && recent_prefetch_queue.issue_prefetch(this, pref_ip, 1))
                        {
				pref_gen_this_cycle++;
                        }
                        processed_flag = 1;
                }

		sjt_present = sjt.find(hash_ip);

                if(!processed_flag && sjt_present)
                {
                        uint64_t sjt_target = sjt.get_target(hash_ip);
                        //Found in SJT, prefetch target, mark processed.
                        if((sjt_target >> (LOG2_BLOCK_SIZE))!= prev_line)
                        {
                                if (L1I_bus.lower_level->get_occupancy(3, 0) < L1I_PQ_SIZE)
                                {
                                        pref_ip = sjt_target;
			
					if(recent_prefetch_queue.issue_prefetch(this,pref_ip, 1))
						pref_gen_this_cycle++;

                                }
                        }
                        processed_flag = 1;
                }

                if(!processed_flag)
                {
                        //Not found anywhere, do next line prefetching and mark processed.
                        if((cur_ip >> (LOG2_BLOCK_SIZE)) != prev_line)
                        {
				if (L1I_bus.lower_level->get_occupancy(3, 0) <L1I_PQ_SIZE && recent_prefetch_queue.issue_prefetch(this, cur_ip, 1)) 
				pref_gen_this_cycle++;
                        }
                        processed_flag = 1;
                }

                //Update prev_line and cur_ip for next iteration.
                prev_line = cur_ip >> (LOG2_BLOCK_SIZE);

                if(pref_ip != 0)
                        cur_ip = pref_ip;
                else
		{
			uint64_t msb = cur_ip >> 16;                
			cur_ip ++;

			if((cur_ip >> 16) != msb)
			{
				cur_ip--;
				uint64_t org_addr = mapper_table.uncompress_addr(cur_ip);
				org_addr++;
				cur_ip = mapper_table.compress_addr(org_addr);
			}
		}

		if(pref_gen_this_cycle > prefetch_degree)
                        break;

	}

        last_prefetch_ip = cur_ip;	
}


uint32_t O3_CPU::prefetcher_cache_fill(uint64_t v_addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_v_addr, uint32_t metadata_in)
{
	return metadata_in;
	
}

void O3_CPU::prefetcher_final_stats()
{
}
