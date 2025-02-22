from abc import ABC, abstractmethod
from typing import List, Set
from src.common.action import Action
from src.common.state import State

class StateUpdateFunction(ABC):
    """Abstract class to define the structure of state update modules. 
    Use `get_new_states` method to get the states to add to RMP."""
    def __init__(self,nodes, actions):
        self.nodes = nodes
        self.actions = actions
    @abstractmethod
    def get_new_states(list_of_actions: List[Action]) -> Set[State]:
        """Generates a set of states to add to `res_states`. Computes this from the `list_of_actions` found in pricing."""
        pass
