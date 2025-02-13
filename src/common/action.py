from typing import List, Dict, Any, Optional, Union
from numpy import ndarray
import uuid


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

    def __init__(self, origin_node: int, destination_node: int, cost: float, contribution_vector: ndarray, trans_min_input: Dict[str, int], trans_term_vec: Dict[str, int], trans_term_min: Dict[str, Union[int, float]]):
        self.origin_node: int = origin_node
        self.destination_node: int = destination_node
        self.cost: float = cost
        self.contribution_vector: ndarray = contribution_vector
        self.trans_min_input: Dict[str, int] = trans_min_input # MIN_RESOURCE_VECTOR
        self.trans_term_vec: Dict[str, int]  = trans_term_vec # RESOURCE_CONSUMPTION_VECTOR
        self.trans_term_min: Dict[str, Union[int, float]] = trans_term_min # MAX_RESOURCE_VECTOR

        self.action_id = uuid.uuid4()


    def is_action_feasible(self, input_resource_vector: Dict[str, int]) -> bool:
        """Checks if `input_resource_vector` meets minimum resource requirements (`trans_min_input`) of this action."""
        return all(input_resource_vector[resource] >= self.trans_min_input[resource]
                   for resource in self.trans_min_input)


    def compute_output_resource_vector(self, input_resource_vector: Dict[str, int]) -> Optional[Dict[str, int]]:
        """Returns the resource vector after taking this action, given an `input_resource_vector`. 
        Finds this by taking input resource (`input_resource_vector[r]`) + resource consumption (`trans_term_vec[r]`), 
        and "dumps" additional resources if this exceeds maximum allowable resource value (`trans_term_min[r]`). 
        Returns `None` if action is infeasible.
        
        
        Arguments:
            input_resource_vector: Dict describing the resource states before taking this action.
        
        """
        if self.is_action_feasible(input_resource_vector) == False:
            return None
        
        output_resource_vector = {}
        for resource in input_resource_vector:
            output_resource_vector[resource] = min(input_resource_vector[resource] + self.trans_term_vec[resource],
                                                    self.trans_term_min[resource])

        return output_resource_vector

    def dominates(self, other: 'Action') -> bool:
         # First condition: cost and ExogVec dominance
        cost_condition = self.cost <= other.cost
        contrib_condition = all(self.contribution_vector >= other.contribution_vector)
        
        if not (cost_condition and contrib_condition):
            return False
            
        # Second condition: strictly better in at least one way
        strict_cost = self.cost < other.cost
        strict_contrib = any(self.contribution_vector[i] > other.contribution_vector[i] 
                            for i in range(len(self.contribution_vector)))
        
        return strict_cost or strict_contrib
    
    def is_null_action(self):
        """
        identifies if current action is a null action
        """
        return (self.origin_node == self.destination_node and
                self.cost == 0.0 and
                all(v == 0 for v in self.contribution_vector))
    

    def __eq__(self, other: 'Action') -> bool:
        """
        Define when two actions are considered equal.
        Two actions are equal if they have the same:
        - origin and destination nodes
        - cost
        - contribution vector
        - resource requirements and constraints
        """
        if not isinstance(other, Action):
            return False
            
        return (
            self.origin_node == other.origin_node and
            self.destination_node == other.destination_node and
            self.cost == other.cost and
            (self.contribution_vector == other.contribution_vector).all() and 
            self.trans_min_input == other.trans_min_input and
            self.trans_term_vec == other.trans_term_vec and
            self.trans_term_min == other.trans_term_min
        )

    def __hash__(self) -> int:
        """
        Define unique hash for action.
        Required when Action objects are used in sets.
        """
        return hash((
            self.origin_node,
            self.destination_node,
            self.cost,
            tuple(self.contribution_vector),
            tuple(sorted(self.trans_min_input.items())),
            tuple(sorted(self.trans_term_vec.items())),
            tuple(sorted(self.trans_term_min.items()))
        ))