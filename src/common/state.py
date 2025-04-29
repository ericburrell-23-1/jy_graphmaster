from random import randint
import hashlib
import numpy as np
from collections import ChainMap
from src.common.helper import Helper
from scipy.sparse import csr_matrix
class State:
    def __init__(self, node:int, state_vec,picked_up, dropped_off,must_dropoff, l_id: int, is_source: bool, is_sink: bool):
        if node == None:
            input('error here for state')
        self.node = node
        self.state_vec = state_vec
        self.picked_up = picked_up
        self.dropped_off = dropped_off
        self.must_drop_off = must_dropoff
        self.l_id=l_id #id for the l in Omega_R.  we can give each graph its own source and sink that does not matter
        self.is_source=is_source #indicates if source
        self.is_sink=is_sink#indicates if sink
        

        self.state_id= hash((self.node,self.is_sink,self.is_source,self.l_id,tuple(self.state_vec),tuple(picked_up),tuple(dropped_off)))


    def __eq__(self, other: 'State') -> bool:
        if other is None:
            return False
        return self.state_id == other.state_id

    def __hash__(self) -> int:
       """
       Provides a hash so that State objects can be used in sets or as dictionary keys.
       We hash by the node and the contents of res_vec.
       """
       return self.state_id

    def this_state_dominates_input_state(self, other_state):
        """
        Determines if this state dominates the input `other_state`.
        Also determines if a tie occurs.
        """
        does_dom = False
        does_equal = False

        # Ensure both states belong to the same node before comparison
        if other_state.node != self.node:
            return [False, False]

        # Convert sparse vectors to dense NumPy arrays to align indices
        vec1_dense = self.state_vec
        vec2_dense = other_state.state_vec
        
        picked_up_2 = other_state.picked_up
        dropped_off_2 = other_state.dropped_off
        # Compute element-wise difference

        res_vec_diff = vec1_dense - vec2_dense
    

        # Compute min and sum values
        min_value = res_vec_diff.min()  # Minimum difference
    
        sum_value = np.abs(res_vec_diff).sum()  # Absolute sum of differences

        # Domination condition
        if min_value >= 0 and sum_value > 0 and picked_up_2==self.picked_up and dropped_off_2.issubset(self.dropped_off):
            does_dom = True

        # Equality check
        if np.array_equal(vec1_dense, vec2_dense) and self.picked_up==picked_up_2 and self.dropped_off==dropped_off_2:
            does_equal = True

        return [does_dom, does_equal]

    def pretty_print_state(self):
        print('state description')
        print('l_id:  '+str(self.l_id))
        print('node:  '+str(self.node))
        print('stateVec:   '+str(self.state_vec.toarray()))
        print('state_id:  '+str(self.state_id))


    def equals(self,secondary_state):
        flag = self.state_id == secondary_state.state_id
        return flag

    def is_source(self):
        return self.node == -1
    def is_sink(self):
        return self.node == -2
    