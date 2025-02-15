import numpy as np
from time import time
from typing import Dict, Set, List, Tuple, Optional
from src.common.state import State
from src.common.action import Action
from src.common.multi_state_graph import MultiStateGraph
from src.algorithm.pricing_problem import PricingProblem
from src.algorithm.update_states.state_update_function import StateUpdateFunction
from src.common.full_multi_graph_object_given_l import Full_Multi_Graph_Object_given_l
from src.common.rmp_graph_given_1 import RMP_graph_given_l
from src.common.pgm_approach import PGM_appraoch

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
        # self.pricer = PricingProblem(
        #     self.actions,
        #     self.initial_resource_state,
        #     self.nodes,
        # )

        self.res_states: Set[State] = initial_res_states
        self.res_states: Set[Action] = initial_res_actions
        self.state_update_function = state_update_module
        #self.multi_graph: Optional[MultiStateGraph] = None
        self.restricted_master_problem = None   
        self.multi_graph = Full_Multi_Graph_Object_given_l(initial_res_states,initial_res_actions,{})
        self.multi_graph.compute_dom_states_by_node()
        self.multi_graph.compute_actions_ub()
        self.pgm_solver = PGM_appraoch(self.multi_graph,self.rhs_exog_vec,initial_res_states,initial_res_actions,None,None)
    def solve(self, max_iterations=1000):
        """
        Solves the GraphMaster problem. This is where the main algorithm will be written.
        """
        if not self.res_paths:
            raise ValueError("Must add initial paths before solving")

        iteration = 1
        incombentLP = -np.inf
        while iteration < max_iterations:
            primal_sol,dual_exog,cur_lp = self.pgm_solver.call_PGM()
            # print("Solving the pricing problem!")
            if cur_lp < incombentLP-1:
                self.pgm_solver.apply_compression_operator()
            # Corrected print statement
            shortest_path, shortest_path_length, ordered_path_rows = self.multi_graph.construct_specific_pricing_pgm(dual_exog)
            #for s1 in states:
                # print(
                #     f"  From state: Node={s1.node}, Resources={s1.resources}")
            # dict_actions = self.convert_actions_to_dict(actions)
            # print(f"actions_in_path = {dict_actions}")
            # print(f"states= {states}")
            if shortest_path_length >= -1e-6:
                return {
                    'status': 'optimal',
                    'x': primal_sol,
                    'iterations': iteration,
                    'graph': self.multi_graph
                }

            self.pgm_solver.apply_expansion_operator(shortest_path, shortest_path_length, ordered_path_rows, self.multi_graph)

            iteration += 1
            self.restricted_master_problem = 0
        return {'status': 'max_iterations', 'iterations': max_iterations}
        
    

        

    def _solve_pricing(self, dual_vector: Dict[int, float]) -> Tuple[List[State], float]:
        """
        Calls the `pricer` solve method using the provided dual vector, and returns the path found.
        This might be redundant by itself. Perhaps replace this method with a `find_new_states` function that calls pricing and the state update function, all-in-one function.
        Also unsure why this function would return a Tuple[List[State], float], so that should probably be fixed.
        """
        # list_of_nodes, list_of_actions, total_cost = self.pricer.generalized_absolute_pricing(dual_vector)
        pass
