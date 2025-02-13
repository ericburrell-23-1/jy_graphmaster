from typing import Dict, Set, List, Tuple, Optional
import numpy as np
from src.common.state import State
from src.common.action import Action
from src.common.multi_state_graph import MultiStateGraph
from src.algorithm.pricing_problem import PricingProblem
from src.algorithm.update_states.state_update_function import StateUpdateFunction


class GraphMaster:
    """
    Entry point to the GraphMaster solver. Takes problem model and initial feasible solution from problem-specific module, and creates the general GraphMaster problem.
    This module will handle the main algorithm, and will make calls to the pricing problem and the restricted master problem. It will also make calls to state update function.

    Problem model consists of:
    - Nodes
    - Actions
    - Exogenous RHS vector
    - Initial Resource State

    Problem-specific modules also need to provide:
    - Initial `res_states`
    - Initial `res_actions`
    - State update function (converts output of pricing problem to new `res_states` and `res_actions`)

    The `solve` method will:
    - Create MultiStateGraph
    - Build and solve RMP
    - Pass dual vector to pricing problem and solve
    - Updates MultiStateGraph and RMP with new `res_states`
    - Repeats until optimal
    - Then solve as ILP
    """

    def __init__(self,
                 actions: Dict[Tuple[int, int], Action],
                 rhs_exog_vec: Dict[int, float],
                 nodes: List[int],
                 initial_resource_state: Dict[str, int],
                 initial_res_states: Set[State],
                 initial_res_actions: Set[Action],
                 state_update_module: StateUpdateFunction
                 ):
        self.nodes = nodes
        self.actions = actions
        self.current_solution = None
        self.rhs_exog_vec = rhs_exog_vec
        self.initial_resource_state = initial_resource_state
        self.pricer = PricingProblem(
            self.actions,
            self.initial_resource_state,
            self.nodes,
        )

        self.res_states: Set[State] = initial_res_states
        self.res_states: Set[Action] = initial_res_actions
        self.state_update_function = state_update_module
        self.multi_graph: Optional[MultiStateGraph] = None
        self.restricted_master_problem = None


    def solve(self, max_iterations=1000):
        """
        Solves the GraphMaster problem. This is where the main algorithm will be written.
        """
        pass

        

    def _solve_pricing(self, dual_vector: Dict[int, float]) -> Tuple[List[State], float]:
        """
        Calls the `pricer` solve method using the provided dual vector, and returns the path found.
        This might be redundant by itself. Perhaps replace this method with a `find_new_states` function that calls pricing and the state update function, all-in-one function.
        Also unsure why this function would return a Tuple[List[State], float], so that should probably be fixed.
        """
        # list_of_nodes, list_of_actions, total_cost = self.pricer.generalized_absolute_pricing(dual_vector)
        pass
