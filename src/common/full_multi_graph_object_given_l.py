from collections import defaultdict
from typing import Any, Dict, Tuple, Set, DefaultDict, List
import networkx as nx
from itertools import permutations
from src.common.helper import Helper
import numpy as np
from src.common.action import Action
from src.common.state import State
from src.common.helper import Helper
from scipy import sparse
import scipy.sparse as sp
import time
from scipy.sparse import vstack
from src.common.time_profile import TimeProfiler
from scipy.sparse import csr_matrix
from tqdm import tqdm
class Full_Multi_Graph_Object_given_l:
 
    #Computed once multi-graph which is generated once
    def __init__(self, l_id, res_states:set[State], all_actions: set[Action],action_dict, dom_actions_pairs,the_null_action,jy_option,input_data):
        """Initializes the object with states, actions, and null action setup."""
        self.l_id = l_id  # ID for the l ∈ Ω_R generating this
        self.rez_states = res_states  # set of all states
        #for state in self.res_states:
        #    print(state.node, state.state_vec.toarray())
        self.all_actions = all_actions  # Set of all possible actions (excluding null action)
        self.dom_actions_pairs = dom_actions_pairs  # Dominating action pairs dictionary
       # self.null_action_info = null_action_info
        self.null_action = the_null_action
        self.action_dict = action_dict
        self.jy_option = jy_option
        #self.resource_name_to_index = resource_name_to_index
        #self.number_of_resources = number_of_resources
        #self.nullAction = self.make_null_action(size_rhs, number_of_resources)  # Create and assign null action
        self.time_profile = defaultdict(float)
        self.data = input_data
        # Initialize dictionary grouping states by node
        with TimeProfiler(self.time_profile, "multi_graph:init_states_by_node"):
            self.resStates_by_node:DefaultDict[int,Set[State]] = defaultdict(set)
            for s in res_states:
                self.resStates_by_node[s.node].add(s)  # Append the actual state object
    
            # Optimized check for source and sink nodes
            node_states = self.resStates_by_node  # Store dictionary lookup once
            source_count = len(node_states.get(-1, []))
            sink_count = len(node_states.get(-2, []))
    
            if source_count != 1 or sink_count != 1:
                raise ValueError(
                    f"Graph {l_id} must have exactly one source and one sink,"
                    f"but found {source_count} source(s) and {sink_count} sink(s)."
                )
            self.source_state=list(self.resStates_by_node[-1])[0]
            self.sink_state=list(self.resStates_by_node[-2])[0]
            self.time_profile = defaultdict(int)


    def LOAD_AI_get_must_drop_off_including_current(self,s):
        NPK=self.jy_option['using_load_ai_lazy_num_pickups']
        NPK=int(NPK)
        self.pickup_node=range(1,NPK+1)
        self.dropoff_node=range(1+NPK,2*NPK)
        drop_off_node_need_to_visit=[]
        if s.node in self.pickup_node:
            #print('part one ')
            drop_off_node_need_to_visit = []
            drop_off_vec = s.state_vec[0,4+len(self.pickup_node):]
            dense_array = drop_off_vec.toarray()[0]
            zero_indices = np.where(dense_array == 0)[0]
            for n in zero_indices:
                drop_off_node_need_to_visit.append(n+1+len(self.dropoff_node))
            drop_off_node_need_to_visit.append(s.node+len(self.dropoff_node))
        elif s.node in self.dropoff_node:
            #print('part two ')

            drop_off_node_need_to_visit = []
            drop_off_vec = s.state_vec[0,4+len(self.pickup_node):]
            dense_array = drop_off_vec.toarray()[0]
            zero_indices = np.where(dense_array == 0)[0]
            for n in zero_indices:
                if n+1+len(self.pickup_node) != s.node:
                    drop_off_node_need_to_visit.append(n+1+len(self.dropoff_node))
        
        # may_avoid_dropoff=s.state_vec[4+self.num_pickups:]
        # must_dropoff=np.nonzero(may_avoid_dropoff<0.5)
        # must_dropoff=may_avoid_dropoff_list+self.num_pickups
        # if s.node in Q.pickup_node:
        #     must_dropoff.append(s.node+self.num_pickups)
        return drop_off_node_need_to_visit

    def make_state_id_to_state(self):
        """Creates a mapping from state ID to state object."""
        with TimeProfiler(self.time_profile, "multi_graph:make_state_id_to_state"):
            self.state_id_to_state = {
                my_state.state_id: my_state
                for node in self.resStates_by_node
                for my_state in self.resStates_by_node[node]
            }
    def debug_action_ub_check(self):
        # remove dominated action for state pair
        
        this_action_dict = defaultdict(list)
        for key,action_list in self.action_dict.items():
            this_action_dict[key]= action_list
        
        for key, action_list in this_action_dict.items():
            remove_action_index = []
            for a1_idx in range(1,len(action_list)):
                for a2_idx in range(len(action_list)-1):
                    if action_list[a1_idx].get_is_dominated(action_list[a2_idx]):
                        remove_action_index.append(a2_idx)
                    if  action_list[a2_idx].get_is_dominated(action_list[a1_idx]):
                        remove_action_index.append(a1_idx)
            this_action_dict[key] = [action_list[i] for i in range(len(action_list)) if i not in remove_action_index]
        # remove dominated states
        this_res_state = self.rez_states.copy()
        # state_to_remove = set()
        # for s1 in self.rez_states:
        #     for s2 in self.rez_states:
        #         if s1.node == s2.node:
        #             does_dom, does_equal = s1.this_state_dominates_input_state(s2)
        #             if does_dom==True and does_equal==False:
        #                 print('s1 dominate s2')
        #                 s1.pretty_print_state()
        #                 s2.pretty_print_state()
        #                 state_to_remove.add(s2)
        # this_res_state.difference_update(state_to_remove)

        # get action ub
        action_ub = defaultdict(set)
        for s1 in this_res_state:
            for s2 in this_res_state:
                if s2 != s1:
                    if s1.node != s2.node:
                        for action in this_action_dict[(s1.node,s2.node)]:
                            if self.is_elementwise_greater_equal(s1.state_vec,action.min_resource_vec) and \
                                self.is_elementwise_greater_equal(self.element_wise_minimum(s1.state_vec + action.resource_consumption_vec,action.max_resource_vec),s2.state_vec):
                                action_ub[(s1,s2)].add(action)
                    else:
                        if self.is_elementwise_strictly_greater(s1.state_vec,s2.state_vec):
                            action_ub[(s1,s2)].add(self.null_action)

        for (s1,s2) in action_ub.keys():
            if s2 == s1:
                input('key error here')
        # action clean
        for(s1,s2), action_list2 in action_ub.items():
            remove_action = set()
            for action in action_list2:
                for (s31,s32), action_list3 in action_ub.items():
                    if s1 == s31 and s2 != s32:
                        is_dom, is_equal = s32.this_state_dominates_input_state(s2)
                        if is_dom and action in action_list3:
                            remove_action.add(action)
                    if s1 != s31 and s2 == s32:
                        is_dom, is_equal = s1.this_state_dominates_input_state(s31)
                        if is_dom and action in action_list3:
                            remove_action.add(action)
                for action2 in action_list2:
                    if action2.get_is_dominated(action) == True:
                        remove_action.add(action)
            action_list2.difference_update(remove_action)
        print('check action ub here')
        # for (s1,s2) ,action_list in action_ub.items():
        #     if (s1,s2) not in self.actions_ub_given_s1s2_2.keys() and len(action_list)>0:
        #         input('key not exsist ')
        #     for action in action_list:
        #         if action not in self.actions_ub_given_s1s2_2[(s1,s2)]:
        #             input(f'action error')
        # Check action_ub against self.actions_ub_given_s1s2_2
        #self.actions_ub_given_s1s2_2
        for (s1, s2), action_list in action_ub.items():
            if len(action_list) > 0:
                if (s1, s2) not in self.actions_s1_s2_clean:
                    print(f"Key {(s1, s2)} exists in action_ub but not in self.actions_ub_given_s1s2_2")
                else:
                    missing_actions = [action for action in action_list if action not in self.actions_s1_s2_clean[(s1, s2)]]
                    if missing_actions:
                        print(f"Missing actions for key {(s1, s2)}: {missing_actions}")

        # Check self.actions_ub_given_s1s2_2 against action_ub
        for (s1, s2), action_list in self.actions_s1_s2_clean.items():
            if len(action_list) > 0:
                if (s1, s2) not in action_ub:
                    print(f"Key {(s1, s2)} exists in self.actions_ub_given_s1s2_2 but not in action_ub")
                else:
                    missing_actions = [action for action in action_list if action not in action_ub[(s1, s2)]]
                    if missing_actions:
                        print(f"Missing actions for key {(s1, s2)}: {missing_actions}")
        
        print('check action ub end')
    def is_elementwise_greater_equal(self,matrix1, matrix2):
        # Convert to dense if matrices are small enough, or use the approach below for larger matrices
        if matrix1.shape != matrix2.shape:
            return False
        
        # Check if matrix1 - matrix2 has any negative elements
        diff = matrix1 - matrix2
        
        # Get minimum value in the difference matrix
        min_value = diff.data.min() if diff.nnz > 0 else 0
        
        # If minimum value is >= 0, then matrix1 is element-wise >= matrix2
        return min_value >= 0
    def element_wise_minimum(self,matrix1, matrix2):
        if matrix1.shape != matrix2.shape:
            raise ValueError("Matrices must have the same shape")
        
        # Convert to COO format for easier manipulation
        A = matrix1.tocoo()
        B = matrix2.tocoo()
        
        # Create dictionaries to store values from both matrices
        values_A = {(i, j): v for i, j, v in zip(A.row, A.col, A.data)}
        values_B = {(i, j): v for i, j, v in zip(B.row, B.col, B.data)}
        
        # Get all positions where either matrix has a non-zero value
        all_positions = set(values_A.keys()) | set(values_B.keys())
        
        # Build the result matrix
        rows, cols, data = [], [], []
        for i, j in all_positions:
            # Get values, defaulting to 0 if position not in matrix
            val_A = values_A.get((i, j), 0)
            val_B = values_B.get((i, j), 0)
            
            # Take the minimum value
            min_val = min(val_A, val_B)
            
            # Only add non-zero values to keep sparsity
            if min_val != 0:
                rows.append(i)
                cols.append(j)
                data.append(min_val)
        
        # Create a new CSR matrix with the minimum values
        result = sparse.csr_matrix((data, (rows, cols)), shape=matrix1.shape)
        
        return result
    def is_elementwise_strictly_greater(self,matrix1, matrix2):
        """
        Check if matrix1 is element-wise strictly greater than matrix2.
        Returns True if all elements in matrix1 > corresponding elements in matrix2.
        
        Parameters:
        -----------
        matrix1, matrix2 : scipy.sparse.csr_matrix
            Sparse matrices to compare
        
        Returns:
        --------
        bool
            True if matrix1 > matrix2 element-wise, False otherwise
        """
        if matrix1.shape != matrix2.shape:
            return False
        
        # Convert matrices to COO format for easier element access
        A = matrix1.tocoo()
        B = matrix2.tocoo()
        
        # Create dictionaries of non-zero elements
        values_A = {(i, j): v for i, j, v in zip(A.row, A.col, A.data)}
        values_B = {(i, j): v for i, j, v in zip(B.row, B.col, B.data)}
        
        # Get all positions where either matrix has a value
        all_positions = set(values_A.keys()) | set(values_B.keys())
        
        for i, j in all_positions:
            # Get values, defaulting to 0 if position not in matrix
            val_A = values_A.get((i, j), 0)
            val_B = values_B.get((i, j), 0)
            
            # If any element in A is not strictly greater than in B, return False
            if val_A <= val_B:
                return False
        
        return True
    def initialize_system(self):
    
        
        
        # Step 1
        
        self.make_state_id_to_state()

        
        # Step 2
        with TimeProfiler(self.time_profile, "multi_graph:compute_actions_ub"):
            #compare_ub_methods_all = self.compare_ub_methods_all()
            self.compute_actions_ub()
            #self.compute_actions_ub_2()
        # Step 3
        with TimeProfiler(self.time_profile, "multi_graph:compute_dom_states_by_node"):
            self.compute_dom_states_by_node()

        
        # Step 4
        with TimeProfiler(self.time_profile, "multi_graph:PGM_sub_compute_min_dominating_states_by_node"):
            self.PGM_sub_compute_min_dominating_states_by_node()

        
        # Step 5
        with TimeProfiler(self.time_profile, "multi_graph:PGM_sub_compute_maximum_dominated_states_by_node"):
            self.PGM_sub_compute_maximum_dominated_states_by_node()

        
        # Step 6
        with TimeProfiler(self.time_profile, "multi_graph:PGM_clean_states_EZ"):
            self.PGM_clean_states_EZ()

        
        # Step 7
        with TimeProfiler(self.time_profile, "multi_graph:PGM_compute_remove_redundant_actions"):
            self.PGM_compute_remove_redundant_actions()

        
        # Step 8
        with TimeProfiler(self.time_profile, "multi_graph:PGM_make_equiv_classes"):
            self.PGM_make_equiv_classes()

        
        # Step 9
        with TimeProfiler(self.time_profile, "multi_graph:construct_pricing_pgm_graph"):
            self.construct_pricing_pgm_graph()

        # step 10 
        self.clean_pricing_pgm_graph()

        # if self.jy_option['debug'] == True:
        #     with TimeProfiler(self.time_profile, "debug"):
        #         self.debug_action_ub_check()
        
        #self.time_profile['multi_graph_initialization'] = time.time() - start_time
        # print("Initialization completed in {:.4f} seconds".format(time.time() - start_time))
        # print("\nExecution time breakdown:")
        # Sort steps by execution time (descending)
        # for step, duration in sorted(self.step_times.items(), key=lambda x: x[1], reverse=True):
        #     print(f"{step}: {duration:.4f} seconds ({duration/sum(self.step_times.values())*100:.1f}%)")
        #input('output multi graph initialization here')
        # if self.jy_option['debug'] == True:
        #     # Step 10
        #     step_start = time.time()
        #     self.debug_action_ub_check()
        #     action_ub_check_time = time.time() - step_start
        #     print(f"action_ub_check: {action_ub_check_time:.4f} seconds")
    
    
    def LOAD_AI_simple_check_valid_action_ub(self,my_action,my_state):
        
        #consider any state $s1$ for which  is at node n1
        # consider the action going from n1 to n2 
        # where n2 is a dropoff node
        # and n1 is a node OTHER THAN THE PICKUP NODE FOR n2 which is called n3
        #suppose that in state s1 that MAY_AVOID_DROPOFF(n2)=1.
        #   this means that n2 has one of the following holds
        #       1.   been visited already in the route. This means taht n1 was picked up and dropped of at n2 prior 
        #       2.   n3 has not been picked up yet
        # in case 1 the path going from state s1 to node n2 dos not make sense.  
        # in case 2 the path going from state s1 to node n2 dos not make sense
        # hence I will not draw the edge.  What this doing  
        LAD=self.jy_option['load_ai_dict']#[]
        node_destination=my_action.node_head
        node_origin=my_action.node_tail
        is_possible=True
        drop_off_node_need_to_visit=self.LOAD_AI_get_must_drop_off_including_current(s)
        if node_destination in LAD['drop_off_nodes'] and node_destination not in drop_off_node_need_to_visit:
            is_possible=False
        if   node_origin!=-1 and len(drop_off_node_need_to_visit)==0 and node_destination!=-2:
            is_possible=False
       # print(is_possible)
       # print('node_destination')
       # print(node_destination)
       # input('hihi')
        return is_possible
    
    def compute_actions_ub_2(self):
        """Computes upper bound actions for each (s1, s2) pair."""
        
        # Initialize defaultdicts properly
        self.actions_ub_given_s1s2_2 = defaultdict(set)
        self.action_tail_head = defaultdict(set)
        self.actions_head_tail = defaultdict(set)
        self.action_ub_tail_head = defaultdict(lambda: defaultdict(set))
        self.action_ub_head_tail = defaultdict(lambda: defaultdict(set))
 
        # Iterate over all actions
        for a1 in self.all_actions:
            node_tail, node_head = a1.node_tail, a1.node_head
            
            for state_tail in self.resStates_by_node[node_tail]:
                head_ideal = a1.get_head_state(state_tail,self.l_id)
                if head_ideal== None:
                    continue
                for state_head in self.resStates_by_node[node_head]:
                    
                    does_dom,does_equal=head_ideal.this_state_dominates_input_state(state_head)
                    
                    if does_dom or does_equal: #head_ideal.this_state_dominates_input_state(state_head): #check if the ideal head dominates the candidate
                        key = (state_tail, state_head)
 
                        # Store results efficiently
                        self.actions_ub_given_s1s2_2[key].add(a1)
                        self.action_ub_tail_head[a1][state_tail].add(state_head)
                        self.action_ub_head_tail[a1][state_head].add(state_tail)
                        a1.check_valid(state_tail, state_head)
    
    def compute_actions_ub(self):
        self.actions_ub_given_s1s2_2 = defaultdict(set)
        self.action_ub_tail_head = defaultdict(lambda: defaultdict(set))
        self.action_ub_head_tail = defaultdict(lambda: defaultdict(set))
    
        for (node_tail,node_head)  in self.action_dict:
            for my_act in self.action_dict[node_tail,node_head]:
                for s1 in self.resStates_by_node[node_tail]:
                    for s2 in self.resStates_by_node[node_head]:
                        #try:
                        is_valid=my_act.check_valid(s1,s2)
                        
                        if is_valid==True:
                            self.actions_ub_given_s1s2_2[(s1,s2)].add(my_act)
                            self.action_ub_tail_head[my_act][s1].add(s2)
                            self.action_ub_head_tail[my_act][s2].add(s1)
                  
                      



    def compute_actions_ub_fast_not_working(self):
        """Computes upper bound actions for each (s1, s2) pair."""
        
        # Precompute dense representations for all states in self.resStates_by_node.
        
        precomputed_dense = {}
        for node, states in self.resStates_by_node.items():
            # Convert to list for consistent indexing
            states_list = list(states)
            if not states_list:
                continue
                
            # Precompute the dense matrix for all states in this bucket.
            candidate_dense = vstack([s.state_vec for s in states_list]).toarray()
            precomputed_dense[node] = (states_list, candidate_dense)
    
        # Initialize defaultdicts properly
        self.actions_ub_given_s1s2_2 = defaultdict(set)
        self.action_tail_head = defaultdict(set)
        self.actions_head_tail = defaultdict(set)
        self.action_ub_tail_head = defaultdict(lambda: defaultdict(set))
        self.action_ub_head_tail = defaultdict(lambda: defaultdict(set))
    
        # Iterate over all actions
        for a1 in tqdm(self.all_actions,desc = 'computing action ub'):
            node_tail, node_head = a1.node_tail, a1.node_head
            
            # Skip if either node doesn't exist in our precomputed data
            if node_head not in precomputed_dense or node_tail not in self.resStates_by_node:
                continue
    
            # Retrieve precomputed candidate data for node_head.
            cand_states_list, cand_dense_all = precomputed_dense[node_head]
            
            # Get all states with matching node values
            for state_tail in self.resStates_by_node[node_tail]:
                with TimeProfiler(self.time_profile, "multi_graph:PGM_make_equiv_classes:get_head_state"):
                    head_ideal = a1.get_head_state(state_tail, self.l_id)
                if head_ideal is None:
                    continue
                if self.jy_option['use_load_ai_in_pgm']==True:
                    if self.LOAD_AI_simple_check_valid_action_ub(a1,state_tail)==False:
                        continue
                # Filter candidate states to only those with the same node as head_ideal
                head_node_candidates = []
                head_node_dense = []
                
                for i, candidate in enumerate(cand_states_list):
                    if candidate.node == head_ideal.node:
                        head_node_candidates.append(candidate)
                        head_node_dense.append(cand_dense_all[i])
                    # else:
                    #     input('chekc here')
                if not head_node_candidates:
                    continue
                    
                # Convert to numpy array for vectorized operations
                head_node_dense = np.array(head_node_dense)
    
                # Convert head_ideal state vector to a dense 1D array.
                head_dense = head_ideal.state_vec.toarray().ravel()
    
                # For proper domination check we need:
                # 1. All elements in head_ideal >= candidate state (min_diff >= 0)
                # 2. Some element in head_ideal > candidate state (sum_diff > 0) 
                # OR states are exactly equal
                
                # Check if head_ideal dominates or equals each candidate
                diff_matrix = head_dense.reshape(1, -1) - head_node_dense
                min_diffs = np.min(diff_matrix, axis=1)
                sum_diffs = np.sum(diff_matrix, axis=1)
                
                # A state dominates if all elements are >= (min_diff >= 0) and sum > 0
                dominates_mask = (min_diffs >= 0) & (sum_diffs > 0)
                
                # Check for equality (all differences are exactly 0)
                equals_mask = np.all(diff_matrix == 0, axis=1)
                
                # Valid candidates are either dominated by or equal to head_ideal
                valid_mask = dominates_mask | equals_mask
                
                # Get the states that are valid based on the mask
                valid_candidates = [candidate for candidate, valid in zip(head_node_candidates, valid_mask) if valid]
    
                # Update dictionaries for all valid candidates
                for candidate in valid_candidates:
                    key = (state_tail, candidate)
                    self.actions_ub_given_s1s2_2[key].add(a1)
                    self.action_ub_tail_head[a1][state_tail].add(candidate)
                    self.action_ub_head_tail[a1][candidate].add(state_tail)
        # debug_on=True
        # if debug_on==True:
        #     self.debug_check_actions_ub()
        

    def debug_check_actions_ub(self):
        self.BACKUP_actions_ub_given_s1s2_2 = defaultdict(set)
        self.BACKUP_action_ub_tail_head = defaultdict(lambda: defaultdict(set))
        self.BACKUP_action_ub_head_tail = defaultdict(lambda: defaultdict(set))
    
        for (node_tail,node_head)  in self.action_dict:
            for my_act in self.action_dict[node_tail,node_head]:
                for s1 in self.resStates_by_node[node_tail]:
                    for s2 in self.resStates_by_node[node_head]:
                        #try:
                        is_valid=my_act.check_valid(s1,s2)
                        
                        if is_valid==True:
                            self.BACKUP_actions_ub_given_s1s2_2[(s1,s2)].add(my_act)
                            self.BACKUP_action_ub_tail_head[my_act][s1].add(s2)
                            self.BACKUP_action_ub_head_tail[my_act][s2].add(s1)
                        
                        if is_valid==True:
                            if s2 not in self.action_ub_tail_head[my_act][s1]:
                                input('error 1')
                            if s1 not in self.action_ub_head_tail[my_act][s2]:
                                input('error 2')
                        else:
                            if s2  in self.action_ub_tail_head[my_act][s1]:
                                input('error 3')
                            if s1  in self.action_ub_head_tail[my_act][s2]:
                                input('error 4')
        self.compare_dictionaries()
        print('check here')
        #is this the same 
        #BACKUP_actions_ub_given_s1s2_2 vs  self.actions_ub_given_s1s2_2
        #BACKUP_action_ub_tail_head vs self.action_ub_tail_head
        #BACKUP_action_ub_head_tail vs self.action_ub_head_tail

    def compute_dom_states_by_node(self):
        #Creates two objects that will be key in the rest of the document
        #state_2_dom_states_dict is a dictionary that when s is put in provdies all states taht s dominates
        #  MEANING s1 in state_2_dom_states_dict:  IFF s1<s
        #state_2_is_dom_states_dict is a dictionary that when s is put in provdies all states that dominate s
        #  MEANING s2 in state_2_dom_states_dict:  IFF s1>s
 
        self.state_2_dom_states_dict = defaultdict(set)
        self.state_2_is_dom_states_dict = defaultdict(set)
    
    # Iterate over all nodes and compute dominance
        print('len(rez_states)')
        print(len(self.rez_states))
        print('starting dom state generation')
        for my_node, states in self.resStates_by_node.items():
            for s1, s2 in permutations(states, 2):  # Generate all ordered pairs (s1, s2)
                if s1.node!=s2.node:
                    input('error here')
                [is_dom,is_equal] = s1.this_state_dominates_input_state(s2)
                if is_dom:
                    self.state_2_dom_states_dict[s1].add(s2)
                    self.state_2_is_dom_states_dict[s2].add(s1)
        print('done state generation')
    def PGM_sub_compute_min_dominating_states_by_node(self):
        #Compute for each s the minimally dominating states .
        #s1 in self.state_min_dom_dict[s] meaning s1>s
        #iff s1 in self.state_2_is_dom_states_dict[s] and no s2 exists s.t.
            #s1 in self.state_2_is_dom_states_dict[s] and s2 in self.state_2_is_dom_states_dict[s1]
            #MENAING  s1>s2 adn s2>s
        self.state_min_dom_dict=dict() #Create a dictionary
        for s in self.state_2_is_dom_states_dict: #itterate over all states
            do_remove=Helper.union_of_sets(self.state_2_is_dom_states_dict,self.state_2_is_dom_states_dict[s]) #compute states to remove
            self.state_min_dom_dict[s]=self.state_2_is_dom_states_dict[s]-do_remove#create object to store states
 
    def PGM_sub_compute_maximum_dominated_states_by_node(self):
        #Compute for each s the maximally dominated states .
 
        #s1 in self.state_max_dom_dict[s]
        #iff s1 in state_2_dom_states_dict and no s2 exists s.t.
            #s2 in self.state_2_dom_states_dict[s] and s1 in self.state_2_dom_states_dict[s2]
            #MENAING  s2<s1 adn s2>s
        self.state_max_dom_dict=dict()#Crate place to store maximally dominated stats
        
        for s in self.state_2_dom_states_dict: #iterate over all states s
            do_remove=Helper.union_of_sets(self.state_2_dom_states_dict,self.state_2_dom_states_dict[s]) #compute states to remove
            self.state_max_dom_dict[s]=self.state_2_dom_states_dict[s]-do_remove#create object to store states
 
    
    def PGM_clean_states_EZ(self):
        #go through all of the states and make srue that only symetrically non-dominated actions are included
        #see the rmp version for details
        
        self.actions_s1_s2_non_dom=defaultdict(set)
 
        for a1 in self.all_actions:
            all_candid_head_given_tail=defaultdict(set)
            all_candid_tail_given_head=defaultdict(set)
            for s_tail in self.action_ub_tail_head[a1]:
                all_heads=self.action_ub_tail_head[a1][s_tail]
                do_remove=Helper.union_of_sets(self.state_max_dom_dict,all_heads)
                all_candid_head_given_tail[s_tail]=all_heads-do_remove
            for s_head in self.action_ub_head_tail[a1]:
                all_tails=self.action_ub_head_tail[a1][s_head]
                do_remove=Helper.union_of_sets(self.state_min_dom_dict,all_tails)
                all_candid_tail_given_head[s_head]=all_tails-do_remove
                tails_to_connect=Helper.subset_where_z_in_Y(s_head,all_candid_tail_given_head[s_head],all_candid_head_given_tail)
                for s_tail in tails_to_connect:
                    if a1 in self.actions_s1_s2_non_dom[(s_tail, s_head)]:
                        input('error here already found')
                    self.actions_s1_s2_non_dom[(s_tail, s_head)].add(a1)
                    #a1.check_valid(s_tail, s_head)
                    #if s_tail.node>0 and s_head.node>0 and s_tail.state_vec.toarray()[0][0]==s_head.state_vec.toarray()[0][0]:
                    ##    print('issue is here too pgm clean')
                    #    input('error here')
    
    def PGM_compute_remove_redundant_actions(self):
        #remove any dominated actions  from each s1,s2
        #see teh rmp vesion for detailss
        self.actions_s1_s2_clean = defaultdict(set)
    
        # Correct syntax: iterate through tuples, not lists
        for (s1, s2) in self.actions_s1_s2_non_dom: 
            my_tup = (s1, s2)
            my_actions = self.actions_s1_s2_non_dom[my_tup] 
            do_remove = Helper.union_of_sets(self.dom_actions_pairs, my_actions)
            self.actions_s1_s2_clean[my_tup] = my_actions - do_remove

            if s1.node > 0 and s2.node > 0 and s1.state_vec.toarray()[0][0] == s2.state_vec.toarray()[0][0]:
                print('issue is here too')
                input('error here')
    #def PGM_make_null_actions(self):  
        #makes null action terms.  This is for dropping resources
        #see the RMP version for this
     #   for s1 in self.state_max_dom_dict:
     #       for s2 in self.state_max_dom_dict[s1]:
     #           this_null_action = Action(self.null_action_info['trans_min_input'],
     #                                     self.null_action_info['trans_term_add'],self.null_action_info['trans_term_min'],
     #                                     s2,s1,self.null_action_info['contribution_vector'],self.null_action_info['cost'],
     #                                     self.null_action_info['min_resource_vec'],self.null_action_info['resource_consumption_vec'],
     #                                     self.null_action_info['indices_non_zero_max'],self.null_action_info['max_resource_vec'])
     #           self.actions_s1_s2_clean[(s1,s2)].add(this_null_action)
     #           self.null_action.add(this_null_action)
 
    def PGM_make_equiv_classes(self):
        #make all equivelence classes
        #see the RMP version for details
        self.equiv_class_2_s1_s2_pairs: DefaultDict[str, set[Tuple[State, State]]] = defaultdict(set) #this will map a number to the s1,s2 pairs that have common action sets
        self.equiv_class_2_actions:DefaultDict[str,Set[Action]]=defaultdict(set) #this will map a number to the s1,s2 pairs that have common action sets
        for [s1,s2] in self.actions_s1_s2_clean: #iterate over s1,s2
            #my_list=[s1.node,s2.node] #create object to store action ids
            my_name_id=[s1.node,s2.node] #create object to store action ids
            my_action_list = []
            for a in self.actions_s1_s2_clean[(s1,s2)]: #store all action ids
                my_action_list.append(a.action_id)
            my_action_list=sorted(my_action_list) #sort the actions ids
            my_name_id.extend(my_action_list)
            my_name_id=str(my_name_id) #convert the action ids to a string
            self.equiv_class_2_s1_s2_pairs[my_name_id].add((s1,s2)) #add the new edge to the equivlenece clas
            if my_name_id not in self.equiv_class_2_actions:
                self.equiv_class_2_actions[my_name_id]=self.actions_s1_s2_clean[(s1,s2)]
    def PGM_equiv_class_dual_2_low_csr(self,action_2_red_cost):
        """Computes the lowest reduced cost action per equivalence class."""
        
        # Compute reduced costs for all actions
        # self.action_2_red_cost = {a1: a1.comp_red_cost(dual_exog_vec) for a1 in self.all_actions}

        # Find the action with the lowest reduced cost per equivalence class
        self.equiv_class_2_low_red_action = {}
        for my_eq_class in self.equiv_class_2_actions:#iterate overs all equivelnce classes
            min_a1 = min(self.equiv_class_2_actions[my_eq_class], key=lambda a1: action_2_red_cost[a1])#copute loewst reduced cost action
            self.equiv_class_2_low_red_action[my_eq_class] = (min_a1, action_2_red_cost[min_a1])# compute the lowest reduced cost action and store the reduced cost
    def PGM_equiv_class_dual_2_low(self, action_2_red_cost):
        """Computes the lowest reduced cost action per equivalence class."""
        
        # Compute reduced costs for all actions
        with TimeProfiler(self.time_profile, "multi_graph:PGM_equiv_class_dual_2_low"):
            
            
            # Find the action with the lowest reduced cost per equivalence class
            self.equiv_class_2_low_red_action = {}

            for my_eq_class, actions in self.equiv_class_2_actions.items():
                min_a1 = None
                min_cost = float('inf')
                
                for a1 in actions:
                    cost = action_2_red_cost[a1]
                    if cost < min_cost:
                        min_cost = cost
                        min_a1 = a1
                        
                self.equiv_class_2_low_red_action[my_eq_class] = (min_a1, min_cost)
            # for my_eq_class in self.equiv_class_2_actions:#iterate overs all equivelnce classes
            #     min_a1 = min(self.equiv_class_2_actions[my_eq_class], key=lambda a1: action_2_red_cost[a1])#copute loewst reduced cost action
            #     self.equiv_class_2_low_red_action[my_eq_class] = (min_a1, action_2_red_cost[min_a1])# compute the lowest reduced cost action and store the reduced cost
    
    def construct_pricing_pgm_graph(self):
        """Constructs the PGM graph with (state_id_tail, state_id_head, equiv_class_id) tuples."""
    
        self.my_rows_pgm_pricing = [
            (s1, s2, eq)
            for eq, pairs in self.equiv_class_2_s1_s2_pairs.items()
            for s1, s2 in pairs
        ]
 


    def construct_specific_pricing_pgm_csr(self, action_2_red_cost,rezStates_minus_by_node):
        """Constructs the PGM pricing graph, computes the shortest path, and extracts the ordered list of rows used."""
        with TimeProfiler(self.time_profile, "multi_graph:construct_specific_pricing_pgm"):
            # Step 1: Compute reduced costs and construct the pricing graph rows
            # with TimeProfiler(self.time_profile, "multi_graph:compute_action_reduced_costs"):
            #     self.compute_action_reduced_costs(dual_exog_vec)
            with TimeProfiler(self.time_profile, "multi_graph:PGM_equiv_class_dual_2_low"):
                self.PGM_equiv_class_dual_2_low_csr(action_2_red_cost)
            with TimeProfiler(self.time_profile, "multi_graph:get_rows_pgm_spec_pricing"):
                self.rows_pgm_spec_pricing = [
                    (row[0].state_id, row[1].state_id, eq_class, action_red_cost, action)
                    for row in self.my_rows_pgm_pricing
                    for eq_class in [row[2]]  # Extract eq_class cleanly
                    for action, action_red_cost in [self.equiv_class_2_low_red_action[eq_class]]  # Unpack action tuple
                ]
            with TimeProfiler(self.time_profile, "multi_graph:construct pricing graph"):
                # Step 2: Create directed graph
                self.pgm_graph = nx.DiGraph()
                #TODO:
                # Step 3: Add edges (tail -> head) with weights (4th index = action_red_cost)
                #print('making graph')
                self.pgm_graph.add_edges_from(
                    (tail, head, {"weight": action_red_cost, "action": action})
                    for tail, head, _, action_red_cost, action in self.rows_pgm_spec_pricing
                )
                #    print(f'node_head:{action.node_head}-{head},node_tail:{action.node_tail}-{tail},weight:{action_red_cost}')
                #    print(f'node_head:{action.node_tail},node_tail:{action.node_head},weight:{action_red_cost}')
                #print('check here')
                #input('----')
                # Step 4: Compute the shortest path from source to sink
                # shortest_path = nx.shortest_path(self.pgm_graph, source=rezStates_minus_by_node[-1].state_id, target=rezStates_minus_by_node[-2].state_id, weight="weight", method="dijkstra")
        
            # # Compute the shortest path cost
            #TODO: call once
            with TimeProfiler(self.time_profile, "multi_graph:looking for shortest path"):
                source, sink = self.source_state.state_id, self.sink_state.state_id
                try:
                    shortest_path_length, shortest_path = nx.single_source_bellman_ford(
                        self.pgm_graph, source=source, target=sink, weight="weight"
                    )
                except nx.NetworkXNoPath:
                    print("No path found from source to sink")
                    return None, float("inf"), []
                # predecessors, distances = nx.bellman_ford_predecessor_and_distance(
                #     self.pgm_graph, 
                #     source=self.source_state.state_id, 
                #     weight="weight"
                # )

                # # Get the shortest path length
                # shortest_path_length = distances[self.sink_state.state_id]

                # # Reconstruct the path
                # shortest_path = [self.sink_state.state_id]
                # current = self.sink_state.state_id
                # while predecessors[current]:  # While current has predecessors
                #     current = predecessors[current][0]  # Take the first predecessor
                #     shortest_path.append(current)
                # shortest_path.reverse()  # Path is built backward, so reverse it
                # shortest_path = nx.bellman_ford_path(self.pgm_graph, source=self.source_state.state_id, target=self.sink_state.state_id, weight="weight")
                # shortest_path_length = nx.bellman_ford_path_length(self.pgm_graph, source=self.source_state.state_id, target=self.sink_state.state_id, weight='weight')
        # shortest_path_length, shortest_path = nx.single_source_dijkstra(self.pgm_graph,
        #                                                   source=self.source_state.state_id ,
        #                                                     target=self.sink_state.state_id ,
        #                                                     weight="weight"
        #                                                 )
            # Step 5: Extract the ordered list of states and actions along the shortest path
            ordered_path_rows = [
                (tail, head, self.pgm_graph[tail][head]["action"])
                for tail, head in zip(shortest_path[:-1], shortest_path[1:])
            ]
        return shortest_path, shortest_path_length, ordered_path_rows
    # def make_null_action(self, size_rhs, size_res_vec):
    #     """Creates a NullAction with zero transitions and no exogenous contribution."""
    #     trans_min_input = np.zeros(size_res_vec)  # Minimum input term
    #     trans_term_add = np.zeros(size_res_vec)  # Addition term
    #     trans_term_min = np.full(size_res_vec, np.inf)  # Minimum transition term
    #     node_tail, node_head = None, None  # No tail or head for null action
    #     action_id = "NullAction"  # Unique identifier for the null action
    #     Exog_vec = np.zeros(size_rhs)  # Exogenous contribution vector
    #     cost = 0  # Null action has no cost
    #     non_zero_indices_exog = []  # Empty since Exog_vec is all zeros
    #     min_resource_vec = np.zeros(size_res_vec)
    #     resource_consumption_vec = np.zeros(size_res_vec)
    #     indices_non_zero_max = []    
    #     max_resource_vec = np.full(size_res_vec, np.inf)
    #     #indices_non_zero_max,max_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,trans_term_min)
    #     return Action(trans_min_input, trans_term_add, trans_term_min, node_tail, node_head, Exog_vec, cost, min_resource_vec,resource_consumption_vec,indices_non_zero_max,max_resource_vec )

    def clean_pricing_pgm_graph(self):

        #make artificial action to reduced cost
        sample_action=list(self.all_actions)[0]
        dual_exog_vec=sample_action.Exog_vec*0
        action_2_red_cost = {a1: a1.comp_red_cost(dual_exog_vec) for a1 in self.all_actions}
        self.PGM_equiv_class_dual_2_low(action_2_red_cost)
        #find unrachable state 
        len_before_edge_set=len(self.my_rows_pgm_pricing)

        UNR_rows_pgm_spec_pricing = [
                    (row[0].state_id, row[1].state_id, eq_class, action_red_cost, action)
                    for row in self.my_rows_pgm_pricing
                    for eq_class in [row[2]]  # Extract eq_class cleanly
                    for action, action_red_cost in [self.equiv_class_2_low_red_action[eq_class]]  # Unpack action tuple
                ]
        G = nx.DiGraph()
        for tail, head, _, action_red_cost, action in UNR_rows_pgm_spec_pricing:
            G.add_edge(tail,head,  weight=action_red_cost, action=action)
        sink_state=self.sink_state.state_id
        source_state=self.source_state.state_id
        
        descendants = nx.descendants(G, source_state)
        #print('descendants')
        #print(descendants)
# Get all nodes that are ancestors of the sink
        ancestors = nx.ancestors(G, sink_state)
        #print('ancestors')
        #print(ancestors)
        common_nodes = descendants.intersection(ancestors)
        common_nodes=common_nodes.union({source_state, sink_state})
        other_nodes = set(G.nodes()) - common_nodes
        # Find the intersection: nodes that are both descendants of the source and ancestors of the sink
        
        self.my_rows_pgm_pricing = [
            row for row in self.my_rows_pgm_pricing
            if row[0].state_id  in common_nodes and row[1].state_id  in common_nodes
        ]
        print('AFTER edge set size')
        len_after_edge_set=len(self.my_rows_pgm_pricing)
        print(len(self.my_rows_pgm_pricing))
        print('len(other_nodes)')
        print(len(other_nodes))
        print('len_before_edge_set')
        print(len_before_edge_set)
        print('len_after_edge_set')
        print(len_after_edge_set)
        input('apres')
        #debug_on=False
        #if debug_on==True:
       #     for sid in other_nodes:
       #         s=self.state_id_to_state[sid]
       #         s.pretty_print_state()
            #if other_nodes
        #if 1==0:
            #if len_before_edge_set!=len_after_edge_set:
                #print('len_before_edge_set')
                #print(len_before_edge_set)
                #print('len_after_edge_set')
                #print(len_after_edge_set)
                #print('self.l_id')
                #print(self.l_id)
                #input('ok not in agreement If i expect this then no good')

                #if self.jy_option['using_load_ai_lazy']==True:
                    #for #sid in other_nodes:
                        #s=self.state_id_to_state[sid]
                        #print('start above no good')
                        ##s.pretty_print_state()
                        #must_drop_off=self.LOAD_AI_get_must_drop_off_including_current(s)

                        #if len(must_drop_off)==0:
                          #  my_act=self.action_dict[s.node,-2][0]
                          #  print('must_drop_off')
                          #  print(must_drop_off)
                          #  print('self.action_dict[s.node,-2][0]')
                          #  print(self.action_dict[s.node,-2][0])
                          #  my_act.pretty_print_action()
                          #  new_head=my_act.get_head_state(s,s.l_id)
                          #  print('new_head')
                          #  print(new_head)
                          #  new_head.pretty_print_state()
                          #  is_valid_one=my_act.check_valid(s,new_head)
                          #  is_valid_two=my_act.check_valid(s,self.sink_state)
                          #  print('is_valid_one')
                          #  print(is_valid_one)
                          #  print('is_valid_two')
                          #  print(is_valid_two)
                          #  print('s in descendants')
                          #  print(s in descendants)
                          #  print('s in ancestors')
                          #  print(s in ancestors)
                          #  input('--state above is no good--')
            #else:
            #    print('OK FINE sizes agree')
            #    print('self.l_id')
            #    print(self.l_id)
            #    input('--')
            #input('num states remove')

            #self.has_done_check_for_unreachable=True
    def construct_specific_pricing_pgm(self, action_2_red_cost,rezStates_minus_by_node):
        """Constructs the PGM pricing graph, computes the shortest path, and extracts the ordered list of rows used."""
        with TimeProfiler(self.time_profile, "multi_graph:construct_specific_pricing_pgm"):
            # Step 1: Compute reduced costs and construct the pricing graph rows
            with TimeProfiler(self.time_profile, "multi_graph:PGM_equiv_class_dual_2_low"):
                self.PGM_equiv_class_dual_2_low(action_2_red_cost)
    
                self.rows_pgm_spec_pricing = [
                    (row[0].state_id, row[1].state_id, eq_class, action_red_cost, action)
                    for row in self.my_rows_pgm_pricing
                    for eq_class in [row[2]]  # Extract eq_class cleanly
                    for action, action_red_cost in [self.equiv_class_2_low_red_action[eq_class]]  # Unpack action tuple
                ]
            with TimeProfiler(self.time_profile, "multi_graph:construct pricing graph"):
                # Step 2: Create directed graph
                self.pgm_graph = nx.DiGraph()
                #TODO:
                # Step 3: Add edges (tail -> head) with weights (4th index = action_red_cost)
                #print('making graph')
                for tail, head, _, action_red_cost, action in self.rows_pgm_spec_pricing:
                    self.pgm_graph.add_edge(tail,head,  weight=action_red_cost, action=action)
                #    print(f'node_head:{action.node_head}-{head},node_tail:{action.node_tail}-{tail},weight:{action_red_cost}')
                #    print(f'node_head:{action.node_tail},node_tail:{action.node_head},weight:{action_red_cost}')
                #print('check here')
                #input('----')
                # Step 4: Compute the shortest path from source to sink
                # shortest_path = nx.shortest_path(self.pgm_graph, source=rezStates_minus_by_node[-1].state_id, target=rezStates_minus_by_node[-2].state_id, weight="weight", method="dijkstra")
        
            # # Compute the shortest path cost
            #TODO: call once
            with TimeProfiler(self.time_profile, "multi_graph:looking for shortest path"):
                predecessors, distances = nx.bellman_ford_predecessor_and_distance(
                    self.pgm_graph, 
                    source=self.source_state.state_id, 
                    weight="weight"
                )

                # Get the shortest path length
                shortest_path_length = distances[self.sink_state.state_id]

                # Reconstruct the path
                shortest_path = [self.sink_state.state_id]
                current = self.sink_state.state_id
                while predecessors[current]:  # While current has predecessors
                    current = predecessors[current][0]  # Take the first predecessor
                    shortest_path.append(current)
                shortest_path.reverse()  # Path is built backward, so reverse it
                # shortest_path = nx.bellman_ford_path(self.pgm_graph, source=self.source_state.state_id, target=self.sink_state.state_id, weight="weight")
                # shortest_path_length = nx.bellman_ford_path_length(self.pgm_graph, source=self.source_state.state_id, target=self.sink_state.state_id, weight='weight')
        # shortest_path_length, shortest_path = nx.single_source_dijkstra(self.pgm_graph,
        #                                                   source=self.source_state.state_id ,
        #                                                     target=self.sink_state.state_id ,
        #                                                     weight="weight"
        #                                                 )
            # Step 5: Extract the ordered list of states and actions along the shortest path
            ordered_path_rows = [
                (tail, head, self.pgm_graph[tail][head]["action"])
                for tail, head in zip(shortest_path[:-1], shortest_path[1:])
            ]
        return shortest_path, shortest_path_length, ordered_path_rows
    def return_time_profile(self):
        return self.time_profile
    def output_profile_time(self):
        for step, duration in sorted(self.time_profile.items(), key=lambda x: x[1], reverse=True):
            print(f"{step}: {duration:.4f} seconds ({duration/sum(self.time_profile.values())*100:.1f}%)")

    def compare_ub_methods_all(self, methods_to_compare=None):
        """
        Compares the outputs of multiple ub computation methods
        to determine if they produce the same dictionaries.
        
        Args:
            methods_to_compare: List of method names to compare. 
                            Default is ["compute_actions_ub", "compute_actions_ub_2", "compute_actions_ub_fast_not_working"]
        
        Returns:
            dict: Dictionary with comparison results for each dictionary and method pair
        """
        if methods_to_compare is None:
            methods_to_compare = [
                "compute_actions_ub",  # Baseline method
                "compute_actions_ub_2",
                "compute_actions_ub_fast_not_working"
            ]
        
        # Store original dictionaries (if they exist)
        original_actions_ub_given_s1s2_2 = getattr(self, 'actions_ub_given_s1s2_2', None)
        original_action_ub_tail_head = getattr(self, 'action_ub_tail_head', None)
        original_action_ub_head_tail = getattr(self, 'action_ub_head_tail', None)
        
        # Dictionary to store results from each method
        method_results = {}
        
        # Run each method and store its results
        for method_name in methods_to_compare:
            if not hasattr(self, method_name):
                print(f"Warning: Method {method_name} does not exist in this class. Skipping.")
                continue
            
            # Reset dictionaries
            self.actions_ub_given_s1s2_2 = defaultdict(set)
            self.action_ub_tail_head = defaultdict(lambda: defaultdict(set))
            self.action_ub_head_tail = defaultdict(lambda: defaultdict(set))
            
            print(f"Running method: {method_name}...")
            try:
                # Get the method and call it
                method = getattr(self, method_name)
                method()
                
                # Store results
                method_results[method_name] = {
                    "actions_ub_given_s1s2_2": self.actions_ub_given_s1s2_2.copy(),
                    "action_ub_tail_head": self.action_ub_tail_head.copy(),
                    "action_ub_head_tail": self.action_ub_head_tail.copy()
                }
                print(f"Successfully ran {method_name}")
            except Exception as e:
                print(f"Error running {method_name}: {str(e)}")
        
        # Restore original dictionaries if they existed
        if original_actions_ub_given_s1s2_2 is not None:
            self.actions_ub_given_s1s2_2 = original_actions_ub_given_s1s2_2
        if original_action_ub_tail_head is not None:
            self.action_ub_tail_head = original_action_ub_tail_head
        if original_action_ub_head_tail is not None:
            self.action_ub_head_tail = original_action_ub_head_tail
        
        # Skip comparison if fewer than 2 methods succeeded
        if len(method_results) < 2:
            print("Not enough methods ran successfully for comparison.")
            return {}
        
        # Compare all methods against the baseline method (first one in the list)
        baseline_method = methods_to_compare[0]
        if baseline_method not in method_results:
            # If baseline failed, use the first successful method as baseline
            baseline_method = list(method_results.keys())[0]
            print(f"Baseline method {methods_to_compare[0]} failed, using {baseline_method} as baseline instead")
        
        comparison_results = {}
        for method_name in methods_to_compare:
            if method_name == baseline_method or method_name not in method_results:
                if method_name not in method_results and method_name != baseline_method:
                    print(f"Skipping comparison with {method_name} as it did not execute successfully")
                continue
            
            print(f"\nComparing {baseline_method} (baseline) vs {method_name}...")
            comparison = self._compare_method_results(
                baseline_method, method_results[baseline_method],
                method_name, method_results[method_name]
            )
            comparison_results[f"{baseline_method}_vs_{method_name}"] = comparison
        
        return comparison_results

    def _compare_method_results(self, method1_name, method1_results, method2_name, method2_results):
        """
        Helper method to compare the results of two methods.
        
        Args:
            method1_name: Name of the first method
            method1_results: Dictionary with results from the first method
            method2_name: Name of the second method
            method2_results: Dictionary with results from the second method
        
        Returns:
            dict: Dictionary with comparison results for each dictionary
        """
        def state_summary(state):
            """Create a readable summary of a State object"""
            return f"State(node={state.node}, id={state.state_id[:8]}..., src={state.is_source}, sink={state.is_sink})"

        def action_summary(action):
            """Create a readable summary of an Action object"""
            return f"Action(id={action.action_id[:8]}..., tail={action.node_tail}, head={action.node_head})"
        
        def summarize_diff(item_type, items, max_display=3):
            """Create a readable summary of a difference"""
            if not items:
                return "None"
            
            result = []
            for i, item in enumerate(items):
                if i >= max_display:
                    result.append(f"... and {len(items) - max_display} more")
                    break
                    
                if item_type == "state":
                    result.append(state_summary(item))
                elif item_type == "action":
                    result.append(action_summary(item))
                elif item_type == "state_pair":
                    result.append(f"({state_summary(item[0])}, {state_summary(item[1])})")
                else:
                    result.append(str(item))
                    
            return ", ".join(result)
        
        # Initialize summary counts for each dictionary
        total_state_pairs = {}
        total_actions = {}
        total_tail_states = {}
        total_head_states = {}
        
        # Detailed comparison results
        results = {
            "actions_ub_given_s1s2_2": {
                "equal": True,
                "differences": [],
                "summary": {}
            },
            "action_ub_tail_head": {
                "equal": True,
                "differences": [],
                "summary": {}
            },
            "action_ub_head_tail": {
                "equal": True,
                "differences": [],
                "summary": {}
            }
        }
        
        # 1. Compare actions_ub_given_s1s2_2
        m1_results = method1_results["actions_ub_given_s1s2_2"]
        m2_results = method2_results["actions_ub_given_s1s2_2"]
        
        m1_keys = set(m1_results.keys())
        m2_keys = set(m2_results.keys())
        
        # Track summary statistics
        total_state_pairs[method1_name] = len(m1_keys)
        total_state_pairs[method2_name] = len(m2_keys)
        
        common_keys = m1_keys.intersection(m2_keys)
        missing_in_m1 = m2_keys - m1_keys
        missing_in_m2 = m1_keys - m2_keys
        
        # Store summary statistics
        results["actions_ub_given_s1s2_2"]["summary"] = {
            "total_state_pairs": {
                method1_name: len(m1_keys),
                method2_name: len(m2_keys)
            },
            "common_state_pairs": len(common_keys),
            "state_pairs_only_in_" + method1_name: len(missing_in_m2),
            "state_pairs_only_in_" + method2_name: len(missing_in_m1)
        }
        
        if m1_keys != m2_keys:
            results["actions_ub_given_s1s2_2"]["equal"] = False
            
            if missing_in_m1:
                message = f"State pairs in {method2_name} but missing in {method1_name}: {len(missing_in_m1)}"
                if len(missing_in_m1) <= 10:
                    message += f" - {summarize_diff('state_pair', missing_in_m1)}"
                results["actions_ub_given_s1s2_2"]["differences"].append(message)
                print(f"Difference detected: {message}")
            
            if missing_in_m2:
                message = f"State pairs in {method1_name} but missing in {method2_name}: {len(missing_in_m2)}"
                if len(missing_in_m2) <= 10:
                    message += f" - {summarize_diff('state_pair', missing_in_m2)}"
                results["actions_ub_given_s1s2_2"]["differences"].append(message)
                print(f"Difference detected: {message}")
        
        # Track action differences for common keys
        action_diff_count = 0
        
        # Check values for common keys
        for key in common_keys:
            m1_actions = m1_results[key]
            m2_actions = m2_results[key]
            
            if m1_actions != m2_actions:
                results["actions_ub_given_s1s2_2"]["equal"] = False
                extra_in_m1 = m1_actions - m2_actions
                extra_in_m2 = m2_actions - m1_actions
                action_diff_count += 1
                
                if extra_in_m1:
                    message = f"For state pair ({state_summary(key[0])}, {state_summary(key[1])}): {len(extra_in_m1)} actions in {method1_name} but not in {method2_name}"
                    if len(extra_in_m1) <= 5:
                        message += f" - {summarize_diff('action', extra_in_m1)}"
                    results["actions_ub_given_s1s2_2"]["differences"].append(message)
                    print(f"Difference detected: {message}")
                
                if extra_in_m2:
                    message = f"For state pair ({state_summary(key[0])}, {state_summary(key[1])}): {len(extra_in_m2)} actions in {method2_name} but not in {method1_name}"
                    if len(extra_in_m2) <= 5:
                        message += f" - {summarize_diff('action', extra_in_m2)}"
                    results["actions_ub_given_s1s2_2"]["differences"].append(message)
                    print(f"Difference detected: {message}")
        
        # Add summary of action differences
        results["actions_ub_given_s1s2_2"]["summary"]["state_pairs_with_different_actions"] = action_diff_count
        
        # 2. Compare action_ub_tail_head
        m1_results = method1_results["action_ub_tail_head"]
        m2_results = method2_results["action_ub_tail_head"]
        
        m1_actions = set(m1_results.keys())
        m2_actions = set(m2_results.keys())
        
        # Track summary statistics
        total_actions[method1_name] = len(m1_actions)
        total_actions[method2_name] = len(m2_actions)
        
        common_actions = m1_actions.intersection(m2_actions)
        missing_in_m1 = m2_actions - m1_actions
        missing_in_m2 = m1_actions - m2_actions
        
        # Store summary statistics
        results["action_ub_tail_head"]["summary"] = {
            "total_actions": {
                method1_name: len(m1_actions),
                method2_name: len(m2_actions)
            },
            "common_actions": len(common_actions),
            "actions_only_in_" + method1_name: len(missing_in_m2),
            "actions_only_in_" + method2_name: len(missing_in_m1)
        }
        
        if m1_actions != m2_actions:
            results["action_ub_tail_head"]["equal"] = False
            
            if missing_in_m1:
                message = f"Actions in {method2_name} but missing in {method1_name}: {len(missing_in_m1)}"
                if len(missing_in_m1) <= 5:
                    message += f" - {summarize_diff('action', missing_in_m1)}"
                results["action_ub_tail_head"]["differences"].append(message)
                print(f"Difference detected: {message}")
            
            if missing_in_m2:
                message = f"Actions in {method1_name} but missing in {method2_name}: {len(missing_in_m2)}"
                if len(missing_in_m2) <= 5:
                    message += f" - {summarize_diff('action', missing_in_m2)}"
                results["action_ub_tail_head"]["differences"].append(message)
                print(f"Difference detected: {message}")
        
        # Track tail state differences
        tail_diff_count = 0
        head_diff_count = 0
        
        # Compare tail states for common actions
        for action in common_actions:
            m1_tails = set(m1_results[action].keys())
            m2_tails = set(m2_results[action].keys())
            
            # Track summary statistics
            if action not in total_tail_states:
                total_tail_states[action] = {}
            total_tail_states[action][method1_name] = len(m1_tails)
            total_tail_states[action][method2_name] = len(m2_tails)
            
            if m1_tails != m2_tails:
                results["action_ub_tail_head"]["equal"] = False
                missing_tails_in_m1 = m2_tails - m1_tails
                missing_tails_in_m2 = m1_tails - m2_tails
                tail_diff_count += 1
                
                if missing_tails_in_m1:
                    message = f"For action {action_summary(action)}: {len(missing_tails_in_m1)} tail states in {method2_name} but missing in {method1_name}"
                    if len(missing_tails_in_m1) <= 5:
                        message += f" - {summarize_diff('state', missing_tails_in_m1)}"
                    results["action_ub_tail_head"]["differences"].append(message)
                    print(f"Difference detected: {message}")
                
                if missing_tails_in_m2:
                    message = f"For action {action_summary(action)}: {len(missing_tails_in_m2)} tail states in {method1_name} but missing in {method2_name}"
                    if len(missing_tails_in_m2) <= 5:
                        message += f" - {summarize_diff('state', missing_tails_in_m2)}"
                    results["action_ub_tail_head"]["differences"].append(message)
                    print(f"Difference detected: {message}")
            
            # Compare head states for each common tail state
            common_tails = m1_tails.intersection(m2_tails)
            for tail in common_tails:
                m1_heads = m1_results[action][tail]
                m2_heads = m2_results[action][tail]
                
                if m1_heads != m2_heads:
                    results["action_ub_tail_head"]["equal"] = False
                    missing_heads_in_m1 = m2_heads - m1_heads
                    missing_heads_in_m2 = m1_heads - m2_heads
                    head_diff_count += 1
                    
                    if missing_heads_in_m1:
                        message = f"For action {action_summary(action)}, tail {state_summary(tail)}: {len(missing_heads_in_m1)} head states in {method2_name} but missing in {method1_name}"
                        if len(missing_heads_in_m1) <= 3:
                            message += f" - {summarize_diff('state', missing_heads_in_m1)}"
                        results["action_ub_tail_head"]["differences"].append(message)
                        print(f"Difference detected: {message}")
                    
                    if missing_heads_in_m2:
                        message = f"For action {action_summary(action)}, tail {state_summary(tail)}: {len(missing_heads_in_m2)} head states in {method1_name} but missing in {method2_name}"
                        if len(missing_heads_in_m2) <= 3:
                            message += f" - {summarize_diff('state', missing_heads_in_m2)}"
                        results["action_ub_tail_head"]["differences"].append(message)
                        print(f"Difference detected: {message}")
        
        # Add summary of tail and head differences
        results["action_ub_tail_head"]["summary"]["actions_with_different_tail_states"] = tail_diff_count
        results["action_ub_tail_head"]["summary"]["tail_states_with_different_head_states"] = head_diff_count
        
        # 3. Compare action_ub_head_tail (similar to action_ub_tail_head but with head/tail swapped)
        m1_results = method1_results["action_ub_head_tail"]
        m2_results = method2_results["action_ub_head_tail"]
        
        m1_actions = set(m1_results.keys())
        m2_actions = set(m2_results.keys())
        
        common_actions = m1_actions.intersection(m2_actions)
        missing_in_m1 = m2_actions - m1_actions
        missing_in_m2 = m1_actions - m2_actions
        
        # Store summary statistics
        results["action_ub_head_tail"]["summary"] = {
            "total_actions": {
                method1_name: len(m1_actions),
                method2_name: len(m2_actions)
            },
            "common_actions": len(common_actions),
            "actions_only_in_" + method1_name: len(missing_in_m2),
            "actions_only_in_" + method2_name: len(missing_in_m1)
        }
        
        if m1_actions != m2_actions:
            results["action_ub_head_tail"]["equal"] = False
            # We don't need to log these differences again as they should be the same as for action_ub_tail_head
        
        # Track head state differences
        head_diff_count = 0
        tail_diff_count = 0
        
        # Compare head states for common actions
        for action in common_actions:
            m1_heads = set(m1_results[action].keys())
            m2_heads = set(m2_results[action].keys())
            
            # Track summary statistics
            if action not in total_head_states:
                total_head_states[action] = {}
            total_head_states[action][method1_name] = len(m1_heads)
            total_head_states[action][method2_name] = len(m2_heads)
            
            if m1_heads != m2_heads:
                results["action_ub_head_tail"]["equal"] = False
                missing_heads_in_m1 = m2_heads - m1_heads
                missing_heads_in_m2 = m1_heads - m2_heads
                head_diff_count += 1
                
                if missing_heads_in_m1:
                    message = f"For action {action_summary(action)}: {len(missing_heads_in_m1)} head states in {method2_name} but missing in {method1_name}"
                    if len(missing_heads_in_m1) <= 5:
                        message += f" - {summarize_diff('state', missing_heads_in_m1)}"
                    results["action_ub_head_tail"]["differences"].append(message)
                    print(f"Difference detected: {message}")
                
                if missing_heads_in_m2:
                    message = f"For action {action_summary(action)}: {len(missing_heads_in_m2)} head states in {method1_name} but missing in {method2_name}"
                    if len(missing_heads_in_m2) <= 5:
                        message += f" - {summarize_diff('state', missing_heads_in_m2)}"
                    results["action_ub_head_tail"]["differences"].append(message)
                    print(f"Difference detected: {message}")
            
            # Compare tail states for each common head state
            common_heads = m1_heads.intersection(m2_heads)
            for head in common_heads:
                m1_tails = m1_results[action][head]
                m2_tails = m2_results[action][head]
                
                if m1_tails != m2_tails:
                    results["action_ub_head_tail"]["equal"] = False
                    missing_tails_in_m1 = m2_tails - m1_tails
                    missing_tails_in_m2 = m1_tails - m2_tails
                    tail_diff_count += 1
                    
                    if missing_tails_in_m1:
                        message = f"For action {action_summary(action)}, head {state_summary(head)}: {len(missing_tails_in_m1)} tail states in {method2_name} but missing in {method1_name}"
                        if len(missing_tails_in_m1) <= 3:
                            message += f" - {summarize_diff('state', missing_tails_in_m1)}"
                        results["action_ub_head_tail"]["differences"].append(message)
                        print(f"Difference detected: {message}")
                    
                    if missing_tails_in_m2:
                        message = f"For action {action_summary(action)}, head {state_summary(head)}: {len(missing_tails_in_m2)} tail states in {method1_name} but missing in {method2_name}"
                        if len(missing_tails_in_m2) <= 3:
                            message += f" - {summarize_diff('state', missing_tails_in_m2)}"
                        results["action_ub_head_tail"]["differences"].append(message)
                        print(f"Difference detected: {message}")
        
        # Add summary of head and tail differences
        results["action_ub_head_tail"]["summary"]["actions_with_different_head_states"] = head_diff_count
        results["action_ub_head_tail"]["summary"]["head_states_with_different_tail_states"] = tail_diff_count
        
        # Generate summary
        all_equal = all(results[key]["equal"] for key in results)
        
        print("\n===== Comparison Summary =====")
        print(f"Comparing {method1_name} vs {method2_name}")
        print(f"All dictionaries equal: {all_equal}")
        for dict_name, result in results.items():
            print(f"\n{dict_name}: {'Equal' if result['equal'] else 'Not Equal'}")
            if not result['equal']:
                print(f"  Summary of differences:")
                for key, value in result["summary"].items():
                    print(f"    {key}: {value}")
                
                print(f"  Found {len(result['differences'])} detailed differences")
                for i, diff in enumerate(result['differences'][:5], 1):  # Show at most 5 differences
                    print(f"    {i}. {diff}")
                if len(result['differences']) > 5:
                    print(f"    ... and {len(result['differences']) - 5} more differences")
        
        return results