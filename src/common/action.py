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

    def __init__(self, trans_min_input:dict, trans_term_add:dict, trans_term_min:dict, 
                node_head:int, node_tail:int, Exog_vec, cost, 
                min_resource_vec:csr_matrix, resource_consumption_vec:csr_matrix, 
                indices_non_zero_max:list, max_resource_vec:csr_matrix, 
                full_resource_vec, empty_resource_vec):
        self.trans_min_input = trans_min_input
        self.trans_term_add = trans_term_add
        self.trans_term_min = trans_term_min
        self.node_tail = node_tail
        self.node_head = node_head
        self.Exog_vec = Exog_vec
        self.Exog_vec_csr = csr_matrix(Exog_vec)
        self.cost = cost
        self.min_resource_vec = min_resource_vec
        if isinstance(self.min_resource_vec, np.ndarray):
            # Reshape to row vector if it's 1D
            self.min_resource_vec = self.min_resource_vec.reshape(1, -1)
            self.min_resource_vec=csr_matrix(self.min_resource_vec)
        self.min_resource_vec_indices = self.min_resource_vec.indices
        self.min_resource_vec_data = self.min_resource_vec.data
        self.resource_consumption_vec = resource_consumption_vec
        self.resource_consumption_indices = self.resource_consumption_vec.indices
        self.resource_consumption_data = self.resource_consumption_vec.data
        self.indices_non_zero_max = indices_non_zero_max
        self.max_resource_vec = max_resource_vec
        self.non_zero_indices_exog = np.nonzero(self.Exog_vec)[0]
        self.full_resource_vec = full_resource_vec
        self.empty_resource_vec = empty_resource_vec
        self.max_vals = {j: int(self.max_resource_vec[0, j]) for j in self.indices_non_zero_max}
        if self.max_vals:
            self.max_indices = np.array(list(self.max_vals.keys()), dtype=int)
            self.max_values = np.array(list(self.max_vals.values()))
        else:
            self.max_indices = np.array([], dtype=int)
            self.max_values = np.array([], dtype=float)
        self.red_cost_non_zero_cal_vals = self.Exog_vec_csr.data
        self.red_cost_non_zero_cal_indices = self.Exog_vec_csr.indices
        self.red_cost_non_zero_cal_indices = np.array(self.red_cost_non_zero_cal_indices, dtype=int)
        # Generate a deterministic hash based on the action's properties
        # Convert numpy arrays to bytes for hashing
        exog_bytes = self.Exog_vec.tobytes() if hasattr(self.Exog_vec, 'tobytes') else str(self.Exog_vec).encode()
        min_res_bytes = self.min_resource_vec.data.tobytes() if hasattr(self.min_resource_vec, 'data') else str(self.min_resource_vec).encode()
        res_cons_bytes = self.resource_consumption_vec.data.tobytes() if hasattr(self.resource_consumption_vec, 'data') else str(self.resource_consumption_vec).encode()
        max_res_bytes = self.max_resource_vec.data.tobytes() if hasattr(self.max_resource_vec, 'data') else str(self.max_resource_vec).encode()
        
        # Create a hash from the combined key properties
        hash_components = [
            str(self.node_tail).encode(),
            str(self.node_head).encode(),
            str(self.cost).encode(),
            exog_bytes,
            min_res_bytes,
            res_cons_bytes,
            str(self.indices_non_zero_max).encode(),
            max_res_bytes
        ]
        
        hash_object = hashlib.md5(b''.join(hash_components))
        self.action_id = hash_object.hexdigest()
        
        self.mark_of_null_action = False
        if self.node_tail is None:
            self.mark_of_null_action = True

    # def comp_red_cost(self, dual_vec):
    #     """Computes the reduced cost by multiplying the dual vector times the exogenous."""
    #     # this_red_cost = 0
    #     # if len(self.non_zero_indices_exog) > 0:
    #     #     this_red_cost =  self.cost - np.dot(self.Exog_vec[self.non_zero_indices_exog], 
    #     #                             dual_vec[self.non_zero_indices_exog])
    #     # else:
    #     #     this_red_cost =  self.cost
    #     this_red_cost_2 =  self.cost - np.sum(self.Exog_vec * dual_vec)

    #     return this_red_cost_2
    def comp_red_cost(self, dual_vec):
        return self.cost - np.dot(self.red_cost_non_zero_cal_vals, dual_vec[self.red_cost_non_zero_cal_indices])
    
    def _get_state_cache_key(self, state_tail: State) -> int:
        """
        Generate a cache key based on the state_tail but excluding l_id.
        """
        # Create a hash of the node, is_source, is_sink, and state_vec, but NOT l_id
        state_vec_hash = tuple(state_tail.state_vec)
        return hash((state_tail.node, state_tail.is_source, state_tail.is_sink, state_vec_hash))

    def get_head_state(self, state_tail: State, l_id):
        """
        Optimized version using zipVec pattern with caching for efficiency.
        """
        # Generate cache key
        cache_key = (self.action_id, self._get_state_cache_key(state_tail))
        
        # Check if we have a cached result
        if cache_key in Action._head_state_cache:
            cached_result = Action._head_state_cache[cache_key]
            # If found in cache, handle accordingly
            if cached_result is None:
                return None
                
            # Unpack the cached result - we store a tuple of (state_vec, is_source, is_sink)
            cached_state_vec, is_source, is_sink = cached_result
            
            # Create a new state with the same data but updated l_id
            return State(self.node_head, cached_state_vec, l_id, is_source, is_sink)
        
        # If not in cache, compute the head state
        # Early checks for minimum resource requirements
        diff_matrix = state_tail.state_vec - self.min_resource_vec
        if np.min(diff_matrix) < 0:
            # Cache the negative result
            Action._head_state_cache[cache_key] = None
            return None
        if diff_matrix.data.size > 0 and np.min(diff_matrix.data) < 0:
            # Cache the negative result
            Action._head_state_cache[cache_key] = None
            return None
            
        # Compute new state vector
        head_state_vec = state_tail.state_vec + self.resource_consumption_vec
        if (head_state_vec.data < 0).any():
            return None
        # Convert to CSR if needed
        if not isinstance(head_state_vec, sp.csr_matrix):
            head_state_vec = head_state_vec.tocsr()
        
        # Create data structures for a new sparse matrix with capped values
        rows = []
        cols = []
        data = []
        
        # Process existing non-zero elements
        cx = head_state_vec.tocoo()  # Convert to COO format for easy iteration
        
        for i, j, v in zip(cx.row, cx.col, cx.data):
            # Apply maximum constraint if needed
            if j in self.indices_non_zero_max:
                max_val = int(self.max_resource_vec[0, j])
                v = min(int(v), max_val)
            
            rows.append(i)
            cols.append(j)
            data.append(v)
        
        # Create new sparse matrix from the processed data
        head_state_vec = sp.csr_matrix((data, (rows, cols)), shape=head_state_vec.shape)
        
        # Create new state object
        if self.node_head == -2:
            this_vec = self.empty_resource_vec
            head_state = State(self.node_head, this_vec, l_id, False, True)
            # Cache the computed head state data
            #Action._head_state_cache[cache_key] = (this_vec, False, True)
        else:
            head_state = State(self.node_head, head_state_vec, l_id, False, False)
            # Cache the computed head state data
            #Action._head_state_cache[cache_key] = (head_state_vec, False, False)
        
        return head_state
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
        if self.violates_min_resources(state_tail.state_vec)==True:
                return None
        # if diff_data.nnz > 0 and (diff_data.data < 0).any():
        #     return None
        # 2. Compute tentative head state vector
        head_state_vec = self._get_head_stat_vec(state_tail)
        # if head_state_vec.nnz > 0 and (head_state_vec.data < 0).any():
        #     return None
        # 3. Apply max_resource cap (only on indices of interest)
        head_state_vec = self.fast_max_res_apply(head_state_vec)
        # if len(self.indices_non_zero_max)>0:
        #     head_state_vec = head_state_vec.tocsr(copy=True)  # Ensure CSR format and not a view
        #     for j in self.indices_non_zero_max:
        #         max_val = int(self.max_resource_vec[0, j])
        #         current_val = head_state_vec[0, j]
        #         if current_val > max_val:
        #             head_state_vec[0, j] = max_val
        time4 = time.time()
        # 4. Create the final State object
        if self.node_head == -2:
            head_state = State(self.node_head, self.empty_resource_vec, l_id, is_source=False, is_sink=True)
        else:
            head_state = State(self.node_head, head_state_vec, l_id, is_source=False, is_sink=False)

        
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
    
    def violates_min_resources(self, tail_vec):   
        
        return np.any(tail_vec[self.min_resource_vec_indices] < self.min_resource_vec_data)

        #return False

    def is_null_action(self):
        """
        identifies if current action is a null action
        """
        return (self.node_tail == self.node_head and
                self.cost == 0.0 and
                all(v == 0 for v in self.Exog_vec))
    
    def __eq__(self, other: "Action") -> bool:
       """
       Checks for equality based on the following fields:
         trans_min_input, trans_term_add, trans_term_min, node_tail,
         node_head, Exog_vec, cost
       """
       if not isinstance(other, Action):
           return False

       return (
           self.trans_min_input == other.trans_min_input and
           self.trans_term_add  == other.trans_term_add  and
           self.trans_term_min  == other.trans_term_min  and
           self.Exog_vec        == other.Exog_vec        and
           self.cost            == other.cost
        )
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
       return hash(self.action_id)
    
    def pretty_print_action(self):
        print('action is ')
        print("self.action_id:  "+str(self.action_id))
        print("self.node_tail:  "+str(self.node_tail))
        print("self.node_head:  "+str(self.node_head))
        print("self.exog_vec:  "+str(self.Exog_vec))

    def check_valid(self, state_tail, state_head):
        is_valid = True

        if self.mark_of_null_action == True and (state_tail.node != state_head.node):
            is_valid = False
            print('not valid due null action for different')
            return is_valid
        if self.mark_of_null_action == False and (state_tail.node != self.node_tail or state_head.node != self.node_head):
            self.pretty_print_action()
            state_head.pretty_print_state()
            state_tail.pretty_print_state()
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