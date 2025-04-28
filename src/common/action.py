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
    """
    Represents an action in the graph.
    Attributes:
        origin_node: The starting node of the action.
        destination_node: The ending node of the action.
        cost: The cost associated with this action.
        contribution_vector: The contribution vector of the action.
        trans_min_input: Dict describing minimum amount of each resource needed for the action to happen.
        trans_term_vec: Dict describing resource consumption.
        trans_term_min: Dict describing the maximum amount of a resource allowed for the action to happen.
    """

    # Class-level caches for head and tail states
    _head_state_cache = {}
    _tail_state_cache = {}

    def __init__(self,node_head:int, node_tail:int,pickup, dropoff, num_cus, Exog_vec, non_zero_exog_vec,cost, 
                min_resource_vec, resource_consumption_vec, max_resource_vec, 
                full_resource_vec, empty_resource_vec):
        self.node_tail = node_tail
        self.node_head = node_head
        self.Exog_vec = Exog_vec
        #self.Exog_vec_csr = csr_matrix(Exog_vec) #1
        self.cost = cost
        self.pickup = pickup
        self.dropoff = dropoff
        self.num_cus = num_cus
        self.min_resource_vec = min_resource_vec
        self.resource_consumption = resource_consumption_vec
        self.max_resource_vec = max_resource_vec
        self.full_resource_vec = full_resource_vec
        self.empty_resource_vec = empty_resource_vec
        self.max_vals = {}
        for idx, val in enumerate(max_resource_vec):
            try:
                self.max_vals[idx] = val
            except:
                print('check here')
        
        if self.max_vals:
            self.max_indices = np.array(list(self.max_vals.keys()), dtype=int)
            self.max_values = np.array(list(self.max_vals.values()))
        else:
            self.max_indices = np.array([], dtype=int)
            self.max_values = np.array([], dtype=float)
        self.red_cost_non_zero_cal_vals = self.red_cost_non_zero_cal_vals = Exog_vec[Exog_vec != 0]
        self.red_cost_non_zero_cal_indices = non_zero_exog_vec
        #self.red_cost_non_zero_cal_indices = np.array(self.red_cost_non_zero_cal_indices, dtype=int)
        self.action_id = hash((self.node_tail,self.node_head))
        
        self.mark_of_null_action = False
        if self.node_tail is None:
            self.mark_of_null_action = True


    def comp_red_cost(self, dual_vec):
        print('check here')
        return self.cost - np.dot(self.red_cost_non_zero_cal_vals, dual_vec[self.red_cost_non_zero_cal_indices])
    

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
        if self.violates_min_resources(state_tail.state_vec, state_tail.picked_up)==True:
            return None
        # if diff_data.nnz > 0 and (diff_data.data < 0).any():
        #     return None
        # 2. Compute tentative head state vector
        head_state_vec = state_tail.state_vec + self.resource_consumption
        # if head_state_vec.nnz > 0 and (head_state_vec.data < 0).any():
        #     return None
        # 3. Apply max_resource cap (only on indices of interest)
        head_state_vec = self.fast_max_res_apply(head_state_vec)
        if self.pickup is not None:
            picked_up = state_tail.picked_up.copy()
            picked_up.add(self.pickup)
            dropped_off = state_tail.dropped_off.copy()
            must_drop_off = state_tail.must_drop_off.copy()
            must_drop_off.add(self.pickup)

        if self.dropoff is not None:
            picked_up = state_tail.picked_up.copy()
            dropped_off = state_tail.dropped_off.copy()
            dropped_off.add(self.dropoff) 
            must_drop_off = state_tail.must_drop_off.copy()
            try:
                must_drop_off.remove(self.dropoff)
            except:
                print('check here')
        if dropped_off.issubset(picked_up) is False:
            print('check here')
        print(picked_up)
        print(dropped_off)

        if self.node_head == -2:
            head_state = State(self.node_head, self.empty_resource_vec,set(),set(),set(), l_id, is_source=False, is_sink=True)
        else:
            head_state = State(self.node_head, head_state_vec,picked_up,dropped_off,must_drop_off, l_id, is_source=False, is_sink=False)

            
        
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

    def get_is_dominated(self, otherAction):
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
    
    def violates_min_resources(self, tail_vec, tail_picked_up):   
        flag = np.any(tail_vec < self.min_resource_vec)
        can_pick_up = True
        can_drop_off = True
        if self.pickup != None:
            can_pick_up = self.pickup not in tail_picked_up
        if self.dropoff != None:
            can_drop_off = (self.dropoff-self.num_cus) in tail_picked_up 
        flag = flag and can_pick_up and  can_drop_off
        return flag


    
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
        # Get current values at the constrained indices
        current_values = head_state_vec_array[self.max_indices]
        
        # Find which values exceed their maximum
        mask = current_values > self.max_values
        
        # Only apply constraints where necessary
        if np.any(mask):
            head_state_vec_array[self.max_indices[mask]] = self.max_values[mask]
        
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
        print("self.exog_vec:  "+str(self.Exog_vec))

    def check_valid(self, state_tail, state_head):
        is_valid = True
        input('it is not in current code')
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
        [is_dom, is_equal] = ideal_head.this_state_dominates_input_state(state_head)
        if is_equal == False and is_dom == False:
           #state_head.pretty_print_state()
           # state_tail.pretty_print_state()
            is_valid = False
           # print('not valid reason 2')
            return is_valid
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