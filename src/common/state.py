from random import randint
from typing import List
class State:
    def __init__(self, node:int, resource_vector, successor_states, predecessor_states, column_index: int = None):
        self.state_id = randint(10**3,10**9)
        self.node = node
        self.resources = resource_vector
        # Please justify why these next two properties exist
        self.successor_states : List[State] = successor_states
        self.predecessor_states : List[State] = predecessor_states
        self.column_index = column_index

    def __eq__(self, other: 'State') -> bool:
        """
        Two states are equal if they have:
        1. Same node
        2. Same capacity remaining
        3. Same can_visit flags for each customer
        """
        if not isinstance(other, State):
            return False
            
        # First check node and capacity
        if (self.node != other.node or 
            self.resources['cap_remain'] != other.resources['cap_remain']):
            return False

        # Then check all can_visit flags
        self_visits = {k: v for k, v in self.resources.items() if k.startswith('can_visit')}
        other_visits = {k: v for k, v in other.resources.items() if k.startswith('can_visit')}
        return self_visits == other_visits

    def __hash__(self) -> int:
        """
        Define unique hash for state that's consistent with equality.
        """
        # Get all can_visit flags in a consistent order
        visit_items = sorted(
            [(k, v) for k, v in self.resources.items() if k.startswith('can_visit')]
        )
        return hash((
            self.node,
            self.resources['cap_remain'],
            tuple(visit_items)
        ))

    def dominates(self, other: 'State') -> bool:
        # s1 dominates s2 if:
        # 1. Same node
        # 2. All resources are greater or equal
        if self.node != other.node:
            return False
        return all(self.resources[r] >= other.resources[r] for r in self.resources)
    def process(self):
        print(f"Processing state: {self.node}")