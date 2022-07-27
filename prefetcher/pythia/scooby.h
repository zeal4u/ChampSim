#ifndef SCOOBY_H
#define SCOOBY_H

#include <vector>
#include <memory>
#include <unordered_map>
#include "champsim.h"
#include "cache.h"
#include "prefetcher.h"
#include "scooby_helper.h"
#include "learning_engine_basic.h"
#include "learning_engine_featurewise.h"
#include "dram_controller.h"

using namespace std;

#define MAX_ACTIONS 64
#define MAX_REWARDS 16
#define MAX_SCOOBY_DEGREE 16
#define SCOOBY_MAX_IPC_LEVEL 4


namespace knob
{
       extern float    scooby_alpha;
       extern float    scooby_gamma;
       extern float    scooby_epsilon;
       extern uint32_t scooby_state_num_bits;
       extern uint32_t scooby_max_states;
       extern uint32_t scooby_seed;
       extern string   scooby_policy;
       extern string   scooby_learning_type;
       extern vector<int32_t> scooby_actions;
       extern uint32_t scooby_max_actions;
       extern uint32_t scooby_pt_size;
       extern uint32_t scooby_st_size;
       extern uint32_t scooby_max_pcs;
       extern uint32_t scooby_max_offsets;
       extern uint32_t scooby_max_deltas;
       extern int32_t  scooby_reward_none;
       extern int32_t  scooby_reward_incorrect;
       extern int32_t  scooby_reward_correct_untimely;
       extern int32_t  scooby_reward_correct_timely;
       extern bool             scooby_brain_zero_init;
       extern bool     scooby_enable_reward_all;
       extern bool     scooby_enable_track_multiple;
       extern bool     scooby_enable_reward_out_of_bounds;
       extern int32_t  scooby_reward_out_of_bounds;
       extern uint32_t scooby_state_type;
       extern bool     scooby_access_debug;
       extern bool     scooby_print_access_debug;
       extern uint64_t scooby_print_access_debug_pc;
       extern uint32_t scooby_print_access_debug_pc_count;
       extern bool     scooby_print_trace;
       extern bool     scooby_enable_state_action_stats;
       extern bool     scooby_enable_reward_tracker_hit;
       extern int32_t  scooby_reward_tracker_hit;
       extern uint32_t scooby_state_hash_type;
       extern bool     scooby_enable_featurewise_engine;
       extern uint32_t scooby_pref_degree;
       extern bool     scooby_enable_dyn_degree;
       extern vector<float> scooby_max_to_avg_q_thresholds;
       extern vector<int32_t> scooby_dyn_degrees;
       extern uint64_t scooby_early_exploration_window;
       extern uint32_t scooby_pt_address_hash_type;
       extern uint32_t scooby_pt_address_hashed_bits;
       extern uint32_t scooby_bloom_filter_size;
       extern uint32_t scooby_multi_deg_select_type;
       extern vector<int32_t> scooby_last_pref_offset_conf_thresholds;
       extern vector<int32_t> scooby_dyn_degrees_type2;
       extern uint32_t scooby_action_tracker_size;
       extern uint32_t scooby_high_bw_thresh;
       extern bool     scooby_enable_hbw_reward;
       extern int32_t  scooby_reward_hbw_correct_timely;
       extern int32_t  scooby_reward_hbw_correct_untimely;
       extern int32_t  scooby_reward_hbw_incorrect;
       extern int32_t  scooby_reward_hbw_none;
       extern int32_t  scooby_reward_hbw_out_of_bounds;
       extern int32_t  scooby_reward_hbw_tracker_hit;
       extern vector<int32_t> scooby_last_pref_offset_conf_thresholds_hbw;
       extern vector<int32_t> scooby_dyn_degrees_type2_hbw;

       /* Learning Engine knobs */
       extern bool     le_enable_trace;
       extern uint32_t le_trace_interval;
       extern std::string   le_trace_file_name;
       extern uint32_t le_trace_state;
       extern bool     le_enable_score_plot;
       extern std::vector<int32_t> le_plot_actions;
       extern std::string   le_plot_file_name;
       extern bool     le_enable_action_trace;
       extern uint32_t le_action_trace_interval;
       extern std::string le_action_trace_name;
       extern bool     le_enable_action_plot;

       /* Featurewise Engine knobs */
       extern vector<int32_t>  le_featurewise_active_features;
       extern vector<int32_t>  le_featurewise_num_tilings;
       extern vector<int32_t>  le_featurewise_num_tiles;
       extern vector<int32_t>  le_featurewise_hash_types;
       extern vector<int32_t>  le_featurewise_enable_tiling_offset;
       extern float                    le_featurewise_max_q_thresh;
       extern bool                             le_featurewise_enable_action_fallback;
       extern vector<float>    le_featurewise_feature_weights;
       extern bool                             le_featurewise_enable_dynamic_weight;
       extern float                    le_featurewise_weight_gradient;
       extern bool                             le_featurewise_disable_adjust_weight_all_features_align;
       extern bool                             le_featurewise_selective_update;
       extern uint32_t                 le_featurewise_pooling_type;
       extern bool             le_featurewise_enable_dyn_action_fallback;
       extern uint32_t                 le_featurewise_bw_acc_check_level;
       extern uint32_t                 le_featurewise_acc_thresh;
       extern bool                     le_featurewise_enable_trace;
       extern uint32_t                 le_featurewise_trace_feature_type;
       extern string                   le_featurewise_trace_feature;
       extern uint32_t                 le_featurewise_trace_interval;
       extern uint32_t                 le_featurewise_trace_record_count;
       extern std::string              le_featurewise_trace_file_name;
       extern bool                     le_featurewise_enable_score_plot;
       extern vector<int32_t>  le_featurewise_plot_actions;
       extern std::string              le_featurewise_plot_file_name;
       extern bool                             le_featurewise_selective_update;
       extern uint32_t                 le_featurewise_pooling_type;
       extern bool             le_featurewise_enable_dyn_action_fallback;
       extern uint32_t                 le_featurewise_bw_acc_check_level;
       extern uint32_t                 le_featurewise_acc_thresh;
       extern bool                     le_featurewise_enable_trace;
       extern uint32_t                 le_featurewise_trace_feature_type;
       extern string                   le_featurewise_trace_feature;
       extern uint32_t                 le_featurewise_trace_interval;
       extern uint32_t                 le_featurewise_trace_record_count;
       extern std::string              le_featurewise_trace_file_name;
       extern bool                     le_featurewise_enable_score_plot;
       extern vector<int32_t>  le_featurewise_plot_actions;
       extern std::string              le_featurewise_plot_file_name;
       extern bool                     le_featurewise_remove_plot_script;
}

/* forward declaration */
class LearningEngine;

class Scooby : public Prefetcher
{
private:
	deque<Scooby_STEntry*> signature_table;
	LearningEngineBasic *brain;
	LearningEngineFeaturewise *brain_featurewise;
	deque<Scooby_PTEntry*> prefetch_tracker;
	Scooby_PTEntry *last_evicted_tracker;
	uint8_t bw_level;
	uint8_t core_ipc;
	uint32_t acc_level;

	/* for workload insights only
	 * has nothing to do with prefetching */
	ScoobyRecorder *recorder;

	/* Data structures for debugging */
	unordered_map<string, uint64_t> target_action_state;

	struct
	{
		struct
		{
			uint64_t lookup;
			uint64_t hit;
			uint64_t evict;
			uint64_t insert;
			uint64_t streaming;
		} st;

		struct
		{
			uint64_t called;
			uint64_t out_of_bounds;
			vector<uint64_t> action_dist;
			vector<uint64_t> issue_dist;
			vector<uint64_t> pred_hit;
			vector<uint64_t> out_of_bounds_dist;
			uint64_t predicted;
			uint64_t multi_deg;
			uint64_t multi_deg_called;
			uint64_t multi_deg_histogram[MAX_SCOOBY_DEGREE+1];
			uint64_t deg_histogram[MAX_SCOOBY_DEGREE+1];
		} predict;

		struct
		{
			uint64_t called;
			uint64_t same_address;
			uint64_t evict;
		} track;

		struct
		{
			struct
			{
				uint64_t called;
				uint64_t pt_not_found;
				uint64_t pt_found;
				uint64_t pt_found_total;
				uint64_t has_reward;
			} demand;

			struct
			{
				uint64_t called;
			} train;

			struct
			{
				uint64_t called;
			} assign_reward;

			struct
			{
				uint64_t dist[MAX_REWARDS][2];
			} compute_reward;
			
			uint64_t correct_timely;
			uint64_t correct_untimely;
			uint64_t no_pref;
			uint64_t incorrect;
			uint64_t out_of_bounds;
			uint64_t tracker_hit;
			uint64_t dist[MAX_ACTIONS][MAX_REWARDS];
		} reward;

		struct
		{
			uint64_t called;
			uint64_t compute_reward;
		} train;

		struct
		{
			uint64_t called;
			uint64_t set;
			uint64_t set_total;
		} register_fill;

		struct
		{
			uint64_t called;
			uint64_t set;
			uint64_t set_total;
		} register_prefetch_hit;

		struct
		{
			uint64_t scooby;
		} pref_issue;

		struct 
		{
			uint64_t epochs;
			uint64_t histogram[DRAM_BW_LEVELS];
		} bandwidth;

		struct 
		{
			uint64_t epochs;
			uint64_t histogram[SCOOBY_MAX_IPC_LEVEL];
		} ipc;

		struct 
		{
			uint64_t epochs;
			uint64_t histogram[CACHE_ACC_LEVELS];
		} cache_acc;
	} stats;

	unordered_map<uint32_t, vector<uint64_t> > state_action_dist;
	unordered_map<std::string, vector<uint64_t> > state_action_dist2;
	unordered_map<int32_t, vector<uint64_t> > action_deg_dist;

private:
	void init_knobs();
	void init_stats();

	void update_global_state(uint64_t pc, uint64_t page, uint32_t offset, uint64_t address);
	Scooby_STEntry* update_local_state(uint64_t pc, uint64_t page, uint32_t offset, uint64_t address);
	uint32_t predict(uint64_t address, uint64_t page, uint32_t offset, shared_ptr<State> &state, vector<uint64_t> &pref_addr);
	bool track(uint64_t address, shared_ptr<State> &state, uint32_t action_index, Scooby_PTEntry **tracker);
	void reward(uint64_t address);
	void reward(Scooby_PTEntry *ptentry);
	void assign_reward(Scooby_PTEntry *ptentry, RewardType type);
	int32_t compute_reward(Scooby_PTEntry *ptentry, RewardType type);
	void train(Scooby_PTEntry *curr_evicted, Scooby_PTEntry *last_evicted);
	vector<Scooby_PTEntry*> search_pt(uint64_t address, bool search_all = false);
	void update_stats(uint32_t state, uint32_t action_index, uint32_t pref_degree = 1);
	void update_stats(shared_ptr<State> &state, uint32_t action_index, uint32_t degree = 1);
	void track_in_st(uint64_t page, uint32_t pred_offset, int32_t pref_offset);
	void gen_multi_degree_pref(uint64_t page, uint32_t offset, int32_t action, uint32_t pref_degree, vector<uint64_t> &pref_addr);
	uint32_t get_dyn_pref_degree(float max_to_avg_q_ratio, uint64_t page = 0xdeadbeef, int32_t action = 0); /* only implemented for CMAC engine 2.0 */
	bool is_high_bw();

public:
	Scooby(string type);
	~Scooby();
	void invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type, vector<uint64_t> &pref_addr);
	void register_fill(uint64_t address);
	void register_prefetch_hit(uint64_t address);
	void dump_stats();
	void print_config();
	int32_t getAction(uint32_t action_index);
	void update_bw(uint8_t bw_level);
	void update_ipc(uint8_t ipc);
	void update_acc(uint32_t acc_level);
};

#endif /* SCOOBY_H */

