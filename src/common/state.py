from random import randint
from typing import List
import numpy as np
import uuid
class State:
    def __init__(self, node:int, state_vec, l_id,is_source,is_sink):
        self.state_id = randint(10**3,10**9)
        self.node = node
        self.state_vec=state_vec #states written out as a vector
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
        return hash((self.node, frozenset(self.res_vec.items())))

    def this_state_dominates_input_state(self,other_state):
        #determined if this state dominates the input other_state
        #also determines if a tie occurs
        does_dom=False #defines default value for domination
        #code for checkign domination between two states
        if other_state.node==self.node and np.min(self.state_vec-self.other_state.state_vec)>=0 and np.sum(self.state_vec-self.other_state.state_vec)>0:
            does_dom=True #set domination to true
        #code for checkign equality  between two statews
        does_equal=False  #defines default value for equality
        if other_state.node==self.node and np.sum(np.abs(self.state_vec-self.other_state.state_vec))==0: #check for equality
            does_equal=True #set domination to false
        return [does_dom,does_equal] #returns the condition
    def process(self):
        print(f"Processing state: {self.node}")