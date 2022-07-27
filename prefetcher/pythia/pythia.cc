#include "cache.h"
#include "champsim.h"
#include "memory_class.h"
#include "scooby.h"
#include "common.h"

vector<shared_ptr<Scooby>> prefetchers;

uint32_t _prefetcher_prefetch_hit(uint32_t cpu, uint64_t addr, uint64_t ip, uint32_t metadata_in)
{
		if(!prefetchers[cpu]->get_type().compare("scooby"))
		{
			shared_ptr<Scooby> pref_scooby = prefetchers[cpu];
			pref_scooby->register_prefetch_hit(addr);
		}
    return metadata_in;
}

void CACHE::prefetcher_initialize()
{
  /**
   * @brief 因为这个对象里含有一些指针，并会动态分配内存，所以你必须这样（new）来创建它。否则
   * 因为vector会深度复制一个对象，如果采用栈上创建，那么Scooby对象会在离开作用域后被释放，包括它的指针所指的对象，但是指针地址却被vector复制了。
   * 更何况这里面有的堆内存并没有被合适的解分配。
   * 
   */
  prefetchers = vector<shared_ptr<Scooby>>(NUM_CPUS, make_shared<Scooby>("scooby"));
  this->prefetcher_prefetch_hit = _prefetcher_prefetch_hit;
}


uint32_t CACHE::prefetcher_cache_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, uint8_t type, uint32_t metadata_in)
{
  // std::cout << "cur_bw_level:" << uint32_t(this->feedback_stat.cur_bw_level)
    // << "cur_ipc:" << uint32_t(this->feedback_stat.cur_ipc)
    // << "acc_level:" << (this->feedback_stat.acc_level) << std::endl;
  vector<uint64_t> pref_addr;
  prefetchers[cpu]->update_bw(this->feedback_stat.cur_bw_level);
  prefetchers[cpu]->update_ipc(this->feedback_stat.cur_ipc);
  prefetchers[cpu]->update_acc(this->feedback_stat.acc_level);
  prefetchers[cpu]->invoke_prefetcher(ip, addr, cache_hit, type, pref_addr);
  for (auto pf_addr : pref_addr)
  {
    prefetch_line(pf_addr, true, 0);
  }
  return metadata_in;
}


uint32_t CACHE::prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in)
{
  return metadata_in;
}

void CACHE::prefetcher_final_stats()
{
}

void CACHE::prefetcher_cycle_operate() {}