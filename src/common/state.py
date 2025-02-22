from random import randint
from typing import List
import numpy as np
import uuid
from collections import ChainMap
from src.common.helper import Helper
import operator
from scipy.sparse import csr_matrix
class State:
    def __init__(self, node:int, state_vec:csr_matrix, l_id,is_source,is_sink):
        #self.state_id = randint(10**3,10**9)
        self.node = node
        if not isinstance(state_vec, csr_matrix):
            raise TypeError("state_vec must be a csr_matrix")
        super().__setattr__('_state_vec', state_vec)
        #self.state_vec=state_vec #states written out as a vector
        # Please justify why these next two properties exist
        self.l_id=l_id #id for the l in Omega_R.  we can give each graph its own source and sink that does not matter
        self.is_source=is_source #indicates if source
        self.is_sink=is_sink#indicates if sink
        self.state_id = uuid.uuid4()
        #self.state_id = state_id

    def __eq__(self, other: 'State') -> bool:
        """
        Two states are equal if they have:
        1. Same node
        2. Same state_vec
        """
        if not isinstance(other, State):
            return False
            
        return (self.node == other.node) and (self.state_vec == other.state_vec) and (self.l_id == other.l_id)

    def __hash__(self) -> int:
        """
        Provides a hash so that State objects can be used in sets or as dictionary keys.
        We hash by the node and the contents of res_vec.
        """
        # Convert res_vec into a frozenset of (key, value) pairs for a hashable representation.
        #return hash((self.node, frozenset(self.res_vec.items())))
        return hash(self.state_id)

    def this_state_dominates_input_state(self, other_state):
        """
        Determines if this state dominates the input `other_state`.
        Also determines if a tie occurs.
        """
        does_dom = False  # Default value for domination

        # Compute difference as a sparse matrix
        res_vec_diff = self.state_vec - other_state.state_vec  # Works for CSR matrices

        # Efficient min and sum operations for sparse matrices
        min_value = res_vec_diff.data.min() if res_vec_diff.nnz > 0 else 0  # Avoids errors if empty
        sum_value = res_vec_diff.sum()  # Works directly on sparse matrices

        # Domination condition
        if other_state.node == self.node and min_value >= 0 and sum_value > 0:
            does_dom = True  # This state dominates

        # Equality check
        does_equal = False  # Default value for equality

        # Correct way to check for zero difference in sparse matrix
        if other_state.node == self.node and res_vec_diff.nnz == 0:  
            does_equal = True  # States are equal

        return [does_dom, does_equal]
    def process(self):
        print(f"Processing state: {self.node}")

    @property
    def state_vec(self):
        return self.__dict__['_state_vec']
    
    def __setattr__(self, name, value):
        # Prevent reassignment of state_vec after initialization.
        if name == 'state_vec' and self.__dict__.get('_initialized', False):
            raise AttributeError("state_vec cannot be changed once set.")
        super().__setattr__(name, value)