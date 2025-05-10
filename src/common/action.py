from typing import List, Dict, Any, Optional, Union, Tuple
from numpy import ndarray
import hashlib
import numpy as np
import operator
from src.common.helper import Helper
from src.common.state import State
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import time
class Action:


    def __init__(self,node_head:int, node_tail:int,pickup, dropoff, num_cus, non_zero_exog_val,non_zero_exog_indices,cost, 
                min_resource_vec, resource_consumption_vec, max_resource_vec,time_window, travel_time, service_time,
                remainder, time_q_rest, time_q1_rest):
        self.node_tail = node_tail
        self.node_head = node_head
        #self.Exog_vec_csr = csr_matrix(Exog_vec) #1
        self.cost = cost
        self.pickup = pickup
        self.dropoff = dropoff
        self.num_cus = num_cus
        self.min_resource_vec = min_resource_vec
        self.resource_consumption = resource_consumption_vec
        self.max_resource_vec = max_resource_vec
        #self.full_resource_vec = full_resource_vec
        self.red_cost_non_zero_cal_vals = non_zero_exog_val
        self.red_cost_non_zero_cal_indices = non_zero_exog_indices
        #self.red_cost_non_zero_cal_indices = np.array(self.red_cost_non_zero_cal_indices, dtype=int)
        self.action_id = hash((self.node_tail,self.node_head))
        self.time_window = time_window
        self.travel_time = travel_time
        self.service_time = service_time
        self.remainder = remainder
        self.time_q_rest = time_q_rest
        self.time_q1_rest = time_q1_rest
        self.mark_of_null_action = False
        if self.node_tail is None:
            self.mark_of_null_action = True


    def comp_red_cost(self, dual_vec):
        if self.red_cost_non_zero_cal_indices != None:
            return self.cost - self.red_cost_non_zero_cal_vals*dual_vec[self.red_cost_non_zero_cal_indices]
        else:
            return self.cost
        #return self.cost - np.dot(self.red_cost_non_zero_cal_vals, dual_vec[self.red_cost_non_zero_cal_indices])
    

    def _get_head_stat_vec(self, state_tail: State):
        head_state_vec = state_tail.state_vec.copy()
        
        # For 1D array, use direct indexing without the first dimension
        for idx, col in enumerate(self.resource_consumption_indices):
            head_state_vec[col] += self.resource_consumption_data[idx]
        
        return head_state_vec
    
    def get_head_state_fast_load_ai(self, state_tail: State, l_id):
        """
        Fast version to compute head state from tail state and resource consumption.
        """

        # 1. Early rejection using sparse comparison (fast & memory efficient)
        #diff_data = state_tail.state_vec - self.min_resource_vec
        #if self.violates_min_resources(state_tail.state_vec)==True:
        violate, earlist_arr_time,new_hos_drive_time, new_hos_work_time = self.hos_violate_and_update_better(state_tail.state_vec)
        if violate == True:
            return None
        
        if self.violates_min_resources(state_tail.state_vec, state_tail.picked_up, state_tail.must_drop_off)==True:
            return None
        
        # if diff_data.nnz > 0 and (diff_data.data < 0).any():
        #     return None
        # 2. Compute tentative head state vector
        head_state_vec = state_tail.state_vec + self.resource_consumption
        head_state_vec[2] = earlist_arr_time
        head_state_vec[4] = new_hos_drive_time
        head_state_vec[5] = new_hos_work_time
        #head_state_vec = self.fast_max_res_apply(head_state_vec)
        picked_up = state_tail.picked_up
        dropped_off = state_tail.dropped_off
        must_drop_off = state_tail.must_drop_off
    
        if self.pickup is not None:
            picked_up = picked_up | {self.pickup}
            must_drop_off = must_drop_off | {self.pickup}
        
        if self.dropoff is not None:
            dropped_off = dropped_off | {self.dropoff}
            # Create a new set for must_drop_off only if we're modifying it
            must_drop_off = must_drop_off - {self.dropoff}
            
        # if np.any(head_state_vec<0):
        #     input('error here Action for statec vec <0 ')
        if self.node_head == -2:
            head_state = State(self.node_head,self.time_window, self.service_time,  np.array([0,0,0,0,0,0]), set(),set(),set(), l_id, is_source=False, is_sink=True)
        else:
            head_state = State(self.node_head, self.time_window, self.service_time, head_state_vec,picked_up,dropped_off,must_drop_off, l_id, is_source=False, is_sink=False)


        do_debug=False
        if do_debug==True:
            backup_head=self.get_head_state(state_tail,state_tail.l_id)
            if (backup_head==None)!=(head_state==None):
                input('error here ')
            if False==backup_head.equals_minus_id(head_state):
                print('error ')
                backup_head.pretty_print_state()
                backup_head.pretty_print_state()
                input('error here ')
            input('GOOD')
        return head_state


    def get_ez_head_state(self, state_tail: State, l_id):
        """
        Fast version to compute head state from tail state and resource consumption.
        """

        # 1. Early rejection using sparse comparison (fast & memory efficient)
        #diff_data = state_tail.state_vec - self.min_resource_vec
        #if self.violates_min_resources(state_tail.state_vec)==True:
        if self.violates_min_resources(state_tail.state_vec, state_tail.picked_up, state_tail.must_drop_off)==True:
            return None
        # if diff_data.nnz > 0 and (diff_data.data < 0).any():
        #     return None
        # 2. Compute tentative head state vector
        head_state_vec = state_tail.state_vec + self.resource_consumption

        head_state_vec = self.fast_max_res_apply(head_state_vec)
        picked_up = state_tail.picked_up
        dropped_off = state_tail.dropped_off
        must_drop_off = state_tail.must_drop_off
    
        if self.pickup is not None:
            picked_up = picked_up | {self.pickup}
            must_drop_off = must_drop_off | {self.pickup}
        
        if self.dropoff is not None:
            dropped_off = dropped_off | {self.dropoff}
            # Create a new set for must_drop_off only if we're modifying it
            must_drop_off = must_drop_off - {self.dropoff}
            

        if self.node_head == -2:
            head_state = State(self.node_head,  self.time_window, self.service_time,np.array([0,0,0,0,0,0]),set(),set(),set(), l_id, is_source=False, is_sink=True)
        else:
            head_state = State(self.node_head, self.time_window, self.service_time, head_state_vec,picked_up,dropped_off,must_drop_off, l_id, is_source=False, is_sink=False)

            
        
        #Handle the case where times are too small to measure
        #print('self.indices_non_zero_max')
        #print(len(self.indices_non_zero_max))

        do_debug=False
        if do_debug==True:
            backup_head=self.get_head_state(state_tail,state_tail.l_id)
            if (backup_head==None)!=(head_state==None):
                input('error here ')
            if False==backup_head.equals_minus_id(head_state):
                print('error ')
                backup_head.pretty_print_state()
                backup_head.pretty_print_state()
                input('error here ')
            input('GOOD')
        return head_state
    def get_tail_state(self, state_head: State, l_id):
        """
        Optimized version of get_tail_state with caching for performance.
        """
        # Generate a cache key that includes action_id and state properties but excludes l_id
        state_vec_hash = tuple(state_head.state_vec)
        cache_key = (self.action_id, hash((state_head.node, state_head.is_source, state_head.is_sink, state_vec_hash)))
        
        # Check if we have a cached result
        if cache_key in Action._tail_state_cache:
            cached_result = Action._tail_state_cache[cache_key]
            # Unpack the cached result - we store a tuple of (state_vec, node, is_source, is_sink)
            cached_state_vec, cached_node, is_source, is_sink = cached_result
            
            # Create a new state with the same data but updated l_id
            return State(cached_node, cached_state_vec, l_id, is_source, is_sink)
        
        # If not in cache, compute the tail state
        # Compute the tail state vector by subtracting the resource consumption
        tail_state_vec = state_head.state_vec - self.resource_consumption_vec
        if not isinstance(tail_state_vec, sp.csr_matrix):
            tail_state_vec = tail_state_vec.tocsr()
            
        # Apply maximum constraints one element at a time to avoid array boolean issues
        for idx in self.indices_non_zero_max:
            # Extract as Python float to avoid array truth value ambiguity
            curr_val = int(tail_state_vec[0, idx])
            max_val = int(self.max_resource_vec[0, idx])
            if curr_val < max_val:  # Changed from max to min since we want to maximize
                tail_state_vec[0, idx] = max_val
        
        # Create the appropriate State object
        if self.node_tail == -1:
            this_vec = self.full_resource_vec
            tail_state = State(self.node_head, this_vec, l_id, True, False)
            
            # Cache the computed tail state data
            Action._tail_state_cache[cache_key] = (this_vec, self.node_head, True, False)
        else:
            tail_state = State(self.node_tail, tail_state_vec, l_id, False, False)
            
            # Cache the computed tail state data
            Action._tail_state_cache[cache_key] = (tail_state_vec, self.node_tail, False, False)
        
        return tail_state

    def OLD_get_is_dominated(self, otherAction):
        """Find out if this action dominates the input action"""
        
        this_dominates_input = False
        if self.node_head != otherAction.node_head or self.node_tail != otherAction.node_tail:
            #please dont comment this out if ogyuant code to be fast
            input('I should not have been called here if you are looping over all actions pairs that is inefficient')
        term_1 = self.cost <= otherAction.cost #find out if the cost is at least as good as input
        max_diff = self.Exog_vec - otherAction.Exog_vec
        term_2 = 0 <= max_diff.max()
        term_1_strict = self.cost < otherAction.cost #find out if the cost is strictly better input
        term_2_strict = np.sum(self.Exog_vec - otherAction.Exog_vec) > 0 #find out if the exogenous is srictly better at some point
        term_3 = term_1_strict or term_2_strict #find out if a strict domination occurs at some point
        if term_1 and term_2 and term_3: #if at least as good and strictly better at some point return true
            this_dominates_input = True #set the domination to true

        return this_dominates_input #return the domination property
    def get_is_dominated(self, otherAction):
        this_dominates_input = False
    
        if self.node_head != otherAction.node_head or self.node_tail != otherAction.node_tail:
            # Please don't comment this out if you want code to be fast
            input('I should not have been called here if you are looping over all actions pairs that is inefficient')
        
        # Check if cost is at least as good
        term_1 = self.cost <= otherAction.cost
        
        # Create dictionaries for fast lookup
        self_dict = dict(zip(self.red_cost_non_zero_cal_indices, self.red_cost_non_zero_cal_vals))
        other_dict = dict(zip(otherAction.red_cost_non_zero_cal_indices, otherAction.red_cost_non_zero_cal_vals))
        
        # Find the maximum difference
        max_diff = float('-inf')
        
        # Check all indices that appear in either vector
        all_indices = set(self.red_cost_non_zero_cal_indices) | set(otherAction.red_cost_non_zero_cal_indices)
        for idx in all_indices:
            self_val = self_dict.get(idx, 0)  # Default to 0 if index not present
            other_val = other_dict.get(idx, 0)  # Default to 0 if index not present
            diff = self_val - other_val
            max_diff = max(max_diff, diff)
        
        term_2 = 0 <= max_diff
        
        # Check for strict dominance in cost
        term_1_strict = self.cost < otherAction.cost
        
        # Check for strict dominance in exogenous vector
        sum_diff = 0
        for idx in all_indices:
            self_val = self_dict.get(idx, 0)
            other_val = other_dict.get(idx, 0)
            diff = self_val - other_val
            sum_diff += diff
        
        term_2_strict = sum_diff > 0
        term_3 = term_1_strict or term_2_strict
        
        if term_1 and term_2 and term_3:
            this_dominates_input = True
        
        return this_dominates_input
    def violates_min_resources(self, tail_vec, tail_picked_up,tail_must_dropoff):   
        if self.pickup != None and self.pickup in tail_picked_up:
            return True
        if self.dropoff != None and self.dropoff not in tail_must_dropoff :
            return True
        if np.any(tail_vec < self.min_resource_vec):
            return True
        return False
    def hos_violate_and_update_better(self,tail_vec):
        # Define constants to improve readability and avoid magic numbers
        MAX_DRIVE = 660
        drive_hos = tail_vec[4]
        
        # Calculate if rest is needed and consolidate the logic
        if self.travel_time <= drive_hos:
            earliest_arr_time = tail_vec[2] - self.travel_time
            new_hos_drive_time = new_hos_work_time= drive_hos - self.remainder
            if earliest_arr_time < self.time_window[1]:
                return True, None, None, None
            return False, earliest_arr_time, new_hos_drive_time, new_hos_work_time
        if drive_hos>=self.remainder:
            earliest_arr_time = tail_vec[2]-self.time_q_rest
            new_hos_drive_time = new_hos_work_time= drive_hos - self.remainder
            if earliest_arr_time < self.time_window[1]:
                return True, None, None, None
            return False, earliest_arr_time, new_hos_drive_time, new_hos_work_time
        earliest_arr_time = tail_vec[2] - self.time_q1_rest
        new_hos_drive_time = new_hos_work_time=MAX_DRIVE-(self.remainder-drive_hos)
        if earliest_arr_time < self.time_window[1]:
                return True, None, None, None
        return False, earliest_arr_time, new_hos_drive_time, new_hos_work_time
    def hos_violate_and_update(self, tail_vec):
        # Define constants to improve readability and avoid magic numbers
        MAX_DRIVE = 660
        REST_TIME = 600
        
        drive_hos = tail_vec[4]
        
        # Calculate if rest is needed and consolidate the logic
        needs_rest = self.travel_time > drive_hos
        
        if needs_rest:
            # Calculate rest periods more efficiently
            excess_time = self.travel_time - drive_hos
            rest_time = (excess_time + MAX_DRIVE - 1) // MAX_DRIVE  # Ceiling division
            
            # Calculate adjusted travel time with rest periods
            travel_time_with_hos = self.travel_time + rest_time * REST_TIME
            
            # Calculate new HOS times (same formula for both drive and work time)
            hos_remaining = drive_hos + rest_time * MAX_DRIVE - self.travel_time
            new_hos_drive_time = new_hos_work_time = hos_remaining
        else:
            # No rest periods needed
            travel_time_with_hos = self.travel_time
            new_hos_drive_time = tail_vec[4] - self.travel_time
            new_hos_work_time = tail_vec[5] - self.travel_time
        
        # Calculate earliest arrival time (same for both branches)
        earliest_arr_time = tail_vec[2] - travel_time_with_hos
        
        # Check if arrival time is within window
        if earliest_arr_time < self.time_window[1]:
            return True, None, None, None
            
        return False, earliest_arr_time, new_hos_drive_time, new_hos_work_time

    def __eq__(self, other: "Action") -> bool:
       """
       Checks for equality based on the following fields:
         trans_min_input, trans_term_add, trans_term_min, node_tail,
         node_head, Exog_vec, cost
       """
       if not isinstance(other, Action):
           return False

       return self.action_id == other.action_id
    
    def fast_max_res_apply(self, head_state_vec_array):
        """
        Apply maximum resource constraints using pre-computed arrays.
        For 1D arrays.
        """

        head_state_vec_array[2] = min(self.max_resource_vec[2],head_state_vec_array[2])
        
        return head_state_vec_array

    def __hash__(self):
       """
       Creates a hash based on the fields used in __eq__.
       """
       return self.action_id
    
    def pretty_print_action(self):
        print('action is ')
        print("self.action_id:  "+str(self.action_id))
        print("self.node_tail:  "+str(self.node_tail))
        print("self.node_head:  "+str(self.node_head))
        print(f'self.red_cost_non_zero_cal_vals:{self.red_cost_non_zero_cal_vals}')
        print(f'self.red_cost_non_zero_cal_indices:{self.red_cost_non_zero_cal_indices}')
        #print("self.exog_vec:  "+str(self.Exog_vec))

    def check_valid(self, state_tail, state_head):
        is_valid = True
        #input('it is not in current code')
        if self.mark_of_null_action == True and (state_tail.node != state_head.node):
            is_valid = False
            print('not valid due null action for different')
            return is_valid
        if self.mark_of_null_action == False and (state_tail.node != self.node_tail or state_head.node != self.node_head):
            self.pretty_print_action()
            #state_head.pretty_print_state()
            #state_tail.pretty_print_state()
            is_valid = False
            print('not valid due to node not agree')
            return is_valid
        ideal_head = self.get_head_state_fast_load_ai(state_tail, state_tail.l_id)
        if ideal_head==None:
            return False
        
        res_vec_diff = state_tail.state_vec[:4] - state_head.state_vec[:4]


        # Compute min and sum values
        min_value = res_vec_diff.min() 
        sum_value = np.abs(res_vec_diff).sum()
        if min_value < 0 or sum_value < 0:
            is_valid == False
        # [is_dom, is_equal] = ideal_head.this_state_dominates_input_state(state_head)
        # if is_equal == False and is_dom == False:
        #    #state_head.pretty_print_state()
        #    # state_tail.pretty_print_state()
        #     is_valid = False
        #    # print('not valid reason 2')
        #     return is_valid
        return is_valid



    # Methods to clear the caches
    @classmethod
    def clear_head_cache(cls):
        """Clear only the head state cache"""
        cls._head_state_cache.clear()
        
    @classmethod
    def clear_tail_cache(cls):
        """Clear only the tail state cache"""
        cls._tail_state_cache.clear()
        
    @classmethod
    def clear_caches(cls):
        """Clear both head and tail state caches"""
        cls._head_state_cache.clear()
        cls._tail_state_cache.clear()