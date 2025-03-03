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
from src.algorithm.gwo_pricing_solver import GWOPricingSolver
from src.common.visulizer import Visulizer
import random
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
                 rhs_exog_vec: np.ndarray,
                 initial_resource_state: Dict[str, int],
                 initial_resource_vector:np.ndarray,
                 initial_res_states: Set[State],
                 initial_res_actions: Set[Action],
                 state_update_module: StateUpdateFunction,
                 initial_dominate_actions:Set[Action],
                 resource_name_to_index: Dict[str, int],
                 number_of_resources: int,
                 the_single_null_action: Action
                 #node_to_list
                 ):
        
        self.nodes = nodes
        self.actions = set().union(*actions.values())
        self.rhs_exog_vec = rhs_exog_vec
        self.initial_resource_state = initial_resource_state
        self.initial_resource_vector = initial_resource_vector
        self.initial_res_states = initial_res_states
        self.initial_res_actions = initial_res_actions
        self.state_update_function = state_update_module
        self.dominate_actions = initial_dominate_actions
        self.resource_name_to_index = resource_name_to_index
        self.number_of_resources = number_of_resources
        self.the_single_null_action=the_single_null_action
        #self.node_to_list = node_to_list
        self.index_to_multi_graph = {}
        self.graph_to_index = {}
        self.rez_states_minus = initial_res_states
        self.res_actions_minus = initial_res_actions
        #self.pricing_problem = PricingProblem(actions,initial_resource_state,nodes, self.resource_name_to_index,initial_resource_vector)
        self.gwo_pricing_solver = GWOPricingSolver(actions,initial_resource_state,nodes, self.resource_name_to_index,initial_resource_vector)
    def solve(self):
        l_id = 0
        random.seed(0)
        size_rhs, size_res_vec = len(self.rhs_exog_vec), len(self.initial_resource_state)
        max_iterations = 100000
        
        my_init_graph=Full_Multi_Graph_Object_given_l(l_id, self.initial_res_states,self.actions, self.dominate_actions,self.the_single_null_action)
        self.rez_states_minus:Set[State]=self.initial_res_states
        self.res_actions=self.initial_res_actions
        #l_id = 0
        #multi_graph = Full_Multi_Graph_Object_given_l(l_id,self.initial_res_states,self.initial_res_actions,self.dominate_actions)
        my_init_graph.initialize_system()
        self.index_to_multi_graph[l_id] = my_init_graph
        iteration = 1
        incombentLP = np.inf
        do_pricing=True
        #print('self.the_single_null_action')
        #print(self.the_single_null_action)
        #input('self.the_single_null_action')
        self.action_id_2_actions={my_action.action_id: my_action for my_action in self.actions}
        debug_init_all_states=False
        debug_init_all_actions=True
        self.lp_before_operations=np.inf
        debug_on=True
        self.complete_routes=[]
        print('starting Graph Master System')
        while iteration < max_iterations:
            
            pgm_solver = PGM_appraoch(self.index_to_multi_graph,self.rhs_exog_vec, self.rez_states_minus,self.res_actions_minus,incombentLP,self.dominate_actions,self.the_single_null_action,self.action_id_2_actions,self.lp_before_operations)
            pgm_solver.call_PGM()
            this_visulizer = Visulizer(pgm_solver)
            this_visulizer.plot_graph()
            print('starting ilp')
            pgm_solver.ilp_solve()
            print('done ilp call')
            self.rez_states_minus, self.res_actions = pgm_solver.return_rez_states_minus_and_res_actions()
            self.complete_routes=pgm_solver.complete_routes
            
            incombentLP = pgm_solver.cur_lp
            self.lp_before_operations=pgm_solver.cur_lp

            l_id += 1
            #all action used in specific column 
            states_used_in_this_col=set([])
            beta_term=[]
            new_states_describing_new_graph=[]
            list_of_actions_used_in_col=set()
            if do_pricing==False:
                #print('in pricing')
                print('in pricing')
                beta_term, new_states_describing_new_graph,states_used_in_this_col = self.state_update_function.get_states_from_random_beta(self.nodes, l_id)
                reduced_cost = -np.inf
            else:
                print('in not  pricing')

                #[list_of_nodes_in_shortest_path, list_of_actions_used_in_col, reduced_cost]= self.pricing_problem.generalized_absolute_pricing(pgm_solver.dual_exog)
                [list_of_nodes_in_shortest_path, list_of_actions_used_in_col, reduced_cost] = self.gwo_pricing_solver.call_gwo_pricing(pgm_solver.dual_exog)
                beta_term, new_states_describing_new_graph,states_used_in_this_col=self.state_update_function.get_new_states(list_of_nodes_in_shortest_path, list_of_actions_used_in_col,l_id)
                print('path')
                print(list_of_nodes_in_shortest_path)
                print('reduce cost')
                print(reduced_cost)
            if reduced_cost >= -1e-6:
                return {
                    'status': 'optimal',
                    'x': pgm_solver.primal_sol,
                    'iterations': iteration,
                    'graph': self.index_to_multi_graph.values()
                }
            print('creating new graph object')
            new_multi_graph = Full_Multi_Graph_Object_given_l(l_id,new_states_describing_new_graph,self.actions,self.dominate_actions,self.the_single_null_action)
            print('initalizing new graph object')

            new_multi_graph.initialize_system()
            print('done initalizing new graph object')
            self.index_to_multi_graph[l_id] = new_multi_graph
            
            #if do_pricing==True:
            if debug_init_all_actions==False:
                #self.res_actions_minus = self.res_actions_minus.union(list_of_actions_used_in_col)
                for my_action in   list_of_actions_used_in_col:
                    self.res_actions_minus.add(my_action)# = self.res_actions_minus.union(list_of_actions_used_in_col)
            else:
                self.res_actions_minus=set()
                for my_action in self.actions:
                    self.res_actions_minus.add(my_action)
            
            if debug_init_all_states==True:
                self.rez_states_minus = self.rez_states_minus.union(new_states_describing_new_graph)
            else:
                #self.res_states_minus = self.res_states_minus.union(states_used_in_this_col)
                for s in states_used_in_this_col:
                    self.rez_states_minus.add(s)

            iteration += 1
            self.restricted_master_problem = 0
            #input(' DONE A COMPLETE GM step')

        return {'status': 'max_iterations', 'iterations': iteration}      
        
    

    

    def _solve_pricing(self, dual_vector: Dict[int, float]) -> Tuple[List[State], float]:
        """
        Calls the `pricer` solve method using the provided dual vector, and returns the path found.
        This might be redundant by itself. Perhaps replace this method with a `find_new_states` function that calls pricing and the state update function, all-in-one function.
        Also unsure why this function would return a Tuple[List[State], float], so that should probably be fixed.
        """
        # list_of_nodes, list_of_actions, total_cost = self.pricer.generalized_absolute_pricing(dual_vector)
        pass
