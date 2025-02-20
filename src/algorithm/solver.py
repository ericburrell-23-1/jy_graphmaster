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
from src.algorithm.update_states.standard_CVRP import CVRP_state_update_function

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
                 nodes: List[int],
                 actions: Dict[Tuple[int, int], Action],
                 rhs_exog_vec: Dict[int, float],
                 initial_resource_state: Dict[str, int],
                 initial_res_states: Set[State],
                 initial_res_actions: Set[Action],
                 state_update_module: StateUpdateFunction,
                 initial_dominate_actions:Set[Action],
                 resource_name_to_index: Dict[str, int],
                 number_of_resources: int
                 #node_to_list
                 ):
        self.nodes = nodes
        self.actions = set().union(*actions.values())
        self.rhs_exog_vec = rhs_exog_vec
        self.initial_resource_state = initial_resource_state
        self.initial_res_states = initial_res_states
        self.initial_res_actions = initial_res_actions
        self.state_update_function = state_update_module
        self.dominate_actions = initial_dominate_actions
        self.resource_name_to_index = resource_name_to_index
        self.number_of_resources = number_of_resources
        #self.node_to_list = node_to_list
        self.index_to_multi_graph = {}
        self.list_of_graph =[]
        self.res_states_minus = initial_res_states
        self.res_actions_minus = initial_res_actions
        self.pricing_problem = PricingProblem(initial_res_actions,initial_res_states,nodes)
        
    def solve(self):
        l_id = 0
        size_rhs, size_res_vec = len(self.rhs_exog_vec), len(self.initial_resource_state)
        max_iterations = 100000
        my_init_graph=Full_Multi_Graph_Object_given_l(l_id, self.initial_res_states,self.actions, self.dominate_actions,size_rhs, self.resource_name_to_index,self.number_of_resources)
        self.res_states_minus=self.initial_res_states
        self.res_actions=self.initial_res_actions
        #l_id = 0
        #multi_graph = Full_Multi_Graph_Object_given_l(l_id,self.initial_res_states,self.initial_res_actions,self.dominate_actions)
        my_init_graph.initialize_system()
        self.index_to_multi_graph[l_id] = my_init_graph
        self.list_of_graph.append(my_init_graph)
        iteration = 1
        incombentLP = np.inf
        while iteration < max_iterations:
            pgm_solver = PGM_appraoch(self.list_of_graph,self.rhs_exog_vec, self.res_states_minus,self.res_actions_minus,incombentLP)
            primal_sol,dual_exog,cur_lp = pgm_solver.call_PGM()
            self.res_states_minus, self.res_actions = pgm_solver.return_rez_states_minus_and_res_actions()
            incombentLP = cur_lp
            #please remember every graph has its own source and sink
            
            [list_of_nodes_in_shortest_path, list_of_actions_used_in_col, reduced_cost]= self.pricing_problem.generalized_absolute_pricing(dual_exog)
            #please remember every graph has its own source and sink
            l_id += 1
            #all action used in specific column 
            [new_states_describing_new_graph,states_used_in_this_col]=self.state_update_function.get_new_states(list_of_nodes_in_shortest_path, list_of_actions_used_in_col,l_id)
            
            if reduced_cost >= -1e-6:
                return {
                    'status': 'optimal',
                    'x': primal_sol,
                    'iterations': iteration,
                    'graph': self.multi_graph
                }
            
            new_multi_graph = Full_Multi_Graph_Object_given_l(l_id,new_states_describing_new_graph,self.actions,self.dominate_actions,size_rhs, self.resource_name_to_index,self.number_of_resources)
            new_multi_graph.initialize_system()
            self.index_to_multi_graph[l_id] = new_multi_graph
            self.list_of_graph.append(new_multi_graph)
            self.res_states_minus = self.res_actions_minus + states_used_in_this_col
            self.res_actions_minus = self.res_actions_minus + list_of_actions_used_in_col
            iteration += 1
            self.restricted_master_problem = 0
        return {'status': 'max_iterations', 'iterations': iteration}      
        
    

    

    def _solve_pricing(self, dual_vector: Dict[int, float]) -> Tuple[List[State], float]:
        """
        Calls the `pricer` solve method using the provided dual vector, and returns the path found.
        This might be redundant by itself. Perhaps replace this method with a `find_new_states` function that calls pricing and the state update function, all-in-one function.
        Also unsure why this function would return a Tuple[List[State], float], so that should probably be fixed.
        """
        # list_of_nodes, list_of_actions, total_cost = self.pricer.generalized_absolute_pricing(dual_vector)
        pass
