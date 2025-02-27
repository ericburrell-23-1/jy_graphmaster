from typing import List, Dict, Any, Optional, Union
from numpy import ndarray
import uuid
import numpy as np
import operator
from src.common.helper import Helper
from src.common.state import State
from scipy.sparse import csr_matrix
class Action:
    """
    Represents an action in the graph.
    Attributes:
        origin_node: The starting node of the action.
        destination_node: The ending node of the action.
        cost: The cost associated with this action.
        contribution_vector: The contribution vector of the action.
        trans_min_input: Dict describing minimum amount of each resource needed for the action to happen. A better name would be `min_resource_vector`.
        trans_term_vec: Dict describing resource consumption. Resource consumption is defined to be negative if a resource is used. A better name would be `resource_consumption_vector`.
        trans_term_min: Dict describing the maximum amount of a resource allowed for the action to happen. A better name would be `max_resource_vector`.

    """

    def __init__(self,trans_min_input:dict,trans_term_add:dict,trans_term_min:dict,node_head:State,node_tail:State,Exog_vec,cost,min_resource_vec:csr_matrix,resource_consumption_vec:csr_matrix,indices_non_zero_max:list,max_resource_vec:csr_matrix):
        self.trans_min_input=trans_min_input #min input term assocatied with an action
        self.trans_term_add=trans_term_add #addition term assocaited with an action
        self.trans_term_min=trans_term_min  #minimum transition term associated with an action 
        self.node_tail=node_tail  #this is the node from which  the action starts at 
        self.node_head=node_head #this is the node from whihc the action ends at 
        #self.action_id=action_id# this is the id associated wit hteh action
        self.Exog_vec=Exog_vec #exogenous contribution vector
        self.cost=cost #cost for the action
        self.min_resource_vec = min_resource_vec
        self.resource_consumption_vec = resource_consumption_vec
        self.indices_non_zero_max = indices_non_zero_max
        self.max_resource_vec = max_resource_vec
        self.non_zero_indices_exog=np.nonzero(self.Exog_vec)[0]
        self.action_id = uuid.uuid4().hex


    def comp_red_cost(self,dual_vec):
        #Computes the reduced cost by mulitplying the dual vector times teh exogenous.  
        
       
        
        return self.cost-np.sum(self.Exog_vec*dual_vec)
    
    def get_head_state(self, state_tail: State, l_id):
        """
        Optimized version of get_head_state that works with NumPy matrix objects
        and avoids unnecessary computations.
        """
        
        # Early return if resource requirements aren't met - using NumPy comparison
        diff_matrix = state_tail.state_vec - self.min_resource_vec
        if np.min(diff_matrix) < 0:
            return None
        
        # Compute new state vector
        head_state_vec = state_tail.state_vec + self.resource_consumption_vec
        
        # Apply maximum constraints one element at a time to avoid array boolean issues
        for idx in self.indices_non_zero_max:
            # Extract as Python float to avoid array truth value ambiguity
            curr_val = float(head_state_vec[0, idx])
            max_val = float(self.max_resource_vec[0, idx])
            head_state_vec[0, idx] = min(curr_val, max_val)
        
        # Create new state object
        if self.node_head == -2:
            head_state = State(self.node_head, head_state_vec, l_id, False, True)
        else:
            head_state = State(self.node_head, head_state_vec, l_id, False, False)
        
        return head_state              
    def get_tail_state(self, state_head: State, l_id):
        """
        Optimized version of get_tail_state that works with NumPy matrix objects
        and corrects the implementation issues from the original function.
        """
        # Compute the tail state vector by subtracting the resource consumption
        tail_state_vec = state_head.state_vec - self.resource_consumption_vec
        
        # Apply maximum constraints one element at a time to avoid array boolean issues
        for idx in self.indices_non_zero_max:
            # Extract as Python float to avoid array truth value ambiguity
            curr_val = float(tail_state_vec[0, idx])
            max_val = float(self.max_resource_vec[0, idx])
            tail_state_vec[0, idx] = max(curr_val, max_val)
        
        # Create the appropriate State object
        if self.node_tail == -1:
            tail_state = State(self.node_tail, tail_state_vec, l_id, True, False)
        else:
            tail_state = State(self.node_tail, tail_state_vec, l_id, False, False)
        
        return tail_state

    def get_is_dominated(self,otherAction):
        #find out if this action dominates the input action 
        this_dominates_input=False
        if self.node_head!=otherAction.node_head or  self.node_tail!=otherAction.node_tail:
            #please dont comment this out if ogyuant code to be fast
            input('I should not have been called here if you are looping over all actions pairs that is inefficient')
        term_1=self.cost<=otherAction.cost #find out if the cost is at least as good as input
        term_2=0<=np.maximum(self.Exog_vec-otherAction.Exog_vec) #find out if this action has at least as good exog vector
        #
        term_1_strict=self.cost<otherAction.cost #find out if the cost is strictly better input
        term_2_strict=np.sum(self.Exog_vec-otherAction.Exog_vec)>0 #find out if the exogenous is srictly better at some point
        term_3=term_1_strict or term_2_strict #find out if a strict domination occurs at some point
        if term_1 and term_2 and term_3: #if at least as good and strictly better at soem point return true
            this_dominates_input=True #set the domination to true

        return this_dominates_input #return the domination property
    
    def is_null_action(self):
        """
        identifies if current action is a null action
        """
        return (self.origin_node == self.destination_node and
                self.cost == 0.0 and
                all(v == 0 for v in self.contribution_vector))
    

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
            self.node_tail       == other.node_tail       and
            self.node_head       == other.node_head       and
            self.Exog_vec        == other.Exog_vec        and
            self.cost            == other.cost
        )

    def __hash__(self):
        """
        Creates a hash based on the fields used in __eq__.
        - Dictionaries are not hashable, so we convert them to frozensets of items.
        """
        return hash(self.action_id)