from numpy import zeros, ones
from src.common.helper import Helper
from scipy.sparse import csr_matrix
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from src.common.state import State
from src.common.action import Action
import numpy as np

class General_state_update:
    def __init__(self, nodes, actions,initial_resource_vector,resource_name_to_index,number_of_resources):
        self.node = nodes
        self.actions = actions
        self.initial_resource_vector = initial_resource_vector
        self.resource_name_to_index = resource_name_to_index
        self.number_of_resources = number_of_resources
        # self.action_to_s1_s2 = defaultdict()
        # for (s1,s2), actions in self.actions.items():
        #     for action in actions:
        #         self.action_to_s1_s2[action] = (s1,s2)


    def state_generation(self, max_depth, depth_used, my_init_states: Set[State], nodes_min_term_vec, actions_reasonable:Set[Action],user_ignore_state_action=None):
        """
        Implementation of Algorithm 1: State Generation Given Pricing
        
        Parameters:
        - max_depth: User input (line 1: MaxDepth ← User)
        - depth_used: User defined function (line 2: DepthUsed(a) ←User Defined)
        - my_init_states: User input from Pricing (line 3: myInitStates ← From User; By Calling Pricing)
        - nodes_min_term_vec: User input (line 4: n.NodeMinTermVec ← User)
        - actions_reasonable: User input (line 5: ActionsReasonable ← User)
        - user_ignore_state_action: Optional user function (line 33: UserIgnoreStateAction(s2, a))
        
        Returns:
        - State2Depth: Dictionary mapping states to their depths
        """
        # Default for user_ignore_state_action if not provided (UserIgnoreStateAction is user-defined, default is False)
        if user_ignore_state_action is None:
            def user_ignore_state_action(state, action):
                # Line 33 in Algorithm 1: UserIgnoreStateAction(s2, a) = True. User defined function; always False by default
                return False
        
        # 6: ActionsSubset ← ActionsReasonable.copy()
        actions_subset = actions_reasonable.copy()
        #my_init_states = my_init_states.copy()
        # 7-11: Add actions between initial states
        for i in range(len(my_init_states)):
            for j in range(i + 1, len(my_init_states)):
                s1 = my_init_states[i]
                s2 = my_init_states[j]
                if (s1.node, s2.node) in self.actions.keys():
                    # Add actions between nodes (in a real implementation, this would come from a function)
                    # This is a placeholder for the actual implementation
                    actions_between = self.actions[(s1.node, s2.node)]
                    actions_subset.union(set(actions_between))
            
        # 12-17: Initialize states that can be expanded
        states_can_expand = []
        for s in my_init_states:
            new_state_vec = self.elementwise_min_csr(s.state_vec, nodes_min_term_vec[s.node])
            
            if np.sum(np.abs(new_state_vec-s.state_vec))>.00001:
                s2 = State(s.node, new_state_vec, s.l_id, s.is_source, s.is_sink)
                states_can_expand.append(s2)
            else:
                states_can_expand.append(s)

        
        # 18: Initialize State2Depth
        state_2_depth = {s: max_depth for s in states_can_expand}
        state_tuple = {(s.node, tuple(s.state_vec.toarray().flatten())) for s in states_can_expand}
        # 19-24: Initialize ActionsFromNode
        actions_from_node = defaultdict(list)
        for a in actions_subset:
            # Check if the action's origin node meets the minimum requirements
            #TODO: element wise
            if self.is_elementwise_greater_equal(nodes_min_term_vec[a.node_tail],a.min_resource_vec):
            #if nodes_min_term_vec[self.action_to_s1_s2[0]] >= a.min_resource_vec:

                actions_from_node[a.node_tail].append(a)

        
        # 25-42: Main loop for state expansion
 
        while len(states_can_expand)>0:
            # 26: Select state with maximum depth
            s = max(states_can_expand, key=lambda x: state_2_depth[x])
            
            # 27: Remove s from states_can_expand
            states_can_expand.remove(s)
            
            # 28-41: Process actions from the current node
            for a in actions_from_node[s.node]:
                # 29: Get next state using the Action's get_head_state method
                
                
                s2 = a.get_head_state(s, s.l_id)
                
                # 30-32: Skip if None (action not valid from this state)
                if s2 is None:
                    continue
                print('====check state update====')
                print('----s1------')
                s.pretty_print_state()
                print('----action------')
                print(a.resource_consumption_vec.toarray())
                print('----s2------')
                s2.pretty_print_state()
                print('============')
                # 33-35: Skip if user_ignore_state_action returns True
                if user_ignore_state_action(s2, a):
                    continue
                    
                # 36: Update s2.stateVec with the minimum values from NodeMinTermVec
                candidate_state_vec= self.elementwise_min_csr(s2.state_vec,nodes_min_term_vec[s2.node])
                if np.array_equal(candidate_state_vec.toarray(), np.array([0,1,0,0])):
                    print('some error here')
                # Check if there are any negative values
                if (candidate_state_vec.data < 0).any():
                    input('some negative in candidate_state_vec')
                if np.sum(np.abs(candidate_state_vec - s2.state_vec)) >0:
                    s2 = State(s2.node, candidate_state_vec, s2.l_id, s2.is_source, s2.is_sink)
                # 37-40: Add to states_can_expand if not seen or has positive depth
                try:
                    #if not self._in_state_dict(s2,state_2_depth) and state_2_depth and state_2_depth[s] > 0:
                    this_key = (s2.node, tuple(s2.state_vec.toarray().flatten()))
                    if this_key not in state_tuple and state_2_depth and state_2_depth[s] > 0:
                        state_2_depth[s2] = state_2_depth[s] - depth_used[a]
                        state_tuple.add(this_key)
                        states_can_expand.append(s2)
                except:
                    print('check this')
        state_2_depth = set(state_2_depth.keys())

        
        
        return state_2_depth

    def _in_state_dict(self, s,state_2_depth):
        for state, depth in state_2_depth.items():
            if state.node == s.node and np.array_equal(state.state_vec.toarray(), s.state_vec.toarray()):
                return True
        return False
    


    def elementwise_min_csr(self,vec1: csr_matrix, vec2: csr_matrix) -> csr_matrix:
        """
        Compute the elementwise minimum of two CSR matrices.
        """
        vec2 = vec2.reshape(1, -1)

        if vec1.shape != vec2.shape:
            raise ValueError(f"Matrices have incompatible shapes: {vec1.shape} vs {vec2.shape}")
        
        # Convert to COO format for easier manipulation
        cx1 = vec1.tocoo()
        cx2 = vec2.tocoo()
        
        # Create dictionaries for non-zero values
        dict1 = {(i, j): v for i, j, v in zip(cx1.row, cx1.col, cx1.data)}
        dict2 = {(i, j): v for i, j, v in zip(cx2.row, cx2.col, cx2.data)}
        
        # Combine keys
        all_keys = set(dict1.keys()).union(set(dict2.keys()))
        
        # Create new data for the minimum values
        rows, cols, data = [], [], []
        for i, j in all_keys:
            # Get values, with 0 as default (not infinity) for missing keys
            val1 = dict1.get((i, j), 0)
            val2 = dict2.get((i, j), 0)
            min_val = min(val1, val2)
            
            # Only include non-zero values in the result
            if min_val != 0:
                rows.append(i)
                cols.append(j)
                data.append(min_val)
        
        # Create a new CSR matrix
        return csr_matrix((data, (rows, cols)), shape=vec1.shape)


    def meets_min_requirements(self,node_min_term_vec: csr_matrix, trans_min_input: dict) -> bool:
        """
        Check if the NodeMinTermVec meets the minimum input requirements for an action.
        
        Parameters:
        - node_min_term_vec: CSR matrix representing the NodeMinTermVec
        - trans_min_input: Dictionary {index: value} representing minimum required values
        
        Returns:
        - True if all minimum requirements are met, False otherwise
        """
        # If trans_min_input is empty, there are no requirements to meet
        if not trans_min_input:
            return True
        
        # Convert to array for easy indexing
        node_min_term_arr = node_min_term_vec.toarray().flatten()
        
        # Check each requirement
        for idx, min_val in trans_min_input.items():
            if idx >= node_min_term_arr.size or node_min_term_arr[idx] < min_val:
                return False
        
        return True

    def get_node_min_term_vec(self,node_id, nodes_min_term_vec):
        """
        Get the NodeMinTermVec for a given node.
        The NodeMinTermVec is directly provided by the user as input.
        
        Parameters:
        - node_id: ID of the node to lookup
        - nodes_min_term_vec: Dictionary mapping node IDs to their NodeMinTermVec (user-provided)
        
        Returns:
        - The NodeMinTermVec for the specified node as a CSR matrix
        """
        # Simply lookup the NodeMinTermVec from the user-provided dictionary
        if node_id in nodes_min_term_vec:
            return nodes_min_term_vec[node_id]
        
        # If not found, return an empty CSR matrix of appropriate size
        # The size should match the expected dimensions in your system
        return csr_matrix((1, 100))  # Assuming dimension of 100, adjust as needed

    def get_all_actions(self,node1_id, node2_id):
        """
        Get all actions from node1 to node2.
        This is a placeholder for the actual implementation.
        """
        # In a real implementation, this would return actions between the specified nodes
        return []

    # The following are placeholder functions for the user-defined components:

    def depth_used(self,action):
        """
        User-defined function to determine depth used by an action.
        As per the document, pickups use 1 and dropoffs use 0.
        
        This is a placeholder - in a real implementation, this would analyze the action
        to determine if it's a pickup or dropoff.
        """
        # Example implementation:
        # 1 for pickups, 0 for dropoffs
        # This is a simplified version based on the document
        if hasattr(action, 'is_pickup') and action.is_pickup:
            return 1
        return 0

    def must_drop_off(state, pickup_id):
        """
        Check if a customer must be dropped off based on state.
        This is a placeholder for the actual implementation.
        """
        # In a real implementation, this would check the state's pickup status
        # This is a simplified example
        return state.state_vec.toarray().flatten()[pickup_id] == 1

    def destination(state):
        """
        Function to get the destination from a state.
        This is a placeholder for the actual implementation.
        """
        # In a real implementation, this might extract destination info from state
        # For now, we just return the node ID
        return state.node

    def user_ignore_state_action(state, action):
        """
        User-defined function to ignore certain state-action pairs.
        As per the document in section 4, this checks for dropping off customers not yet picked up.
        """
        # Get the pickup ID from the action's destination
        pickup_id = action.node_head  # Assuming the node ID is the pickup ID
        
        # If trying to drop off a customer not yet picked up
        if not must_drop_off(state, pickup_id) and destination(state) == pickup_id:
            return True
        return False
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