import numpy as np
from typing import Dict, Set, List, Tuple, Optional
from src.common.state import State
from src.common.action import Action
from src.algorithm.pricing_problem import PricingProblem
from src.algorithm.update_states.state_update_function import StateUpdateFunction
from src.common.full_multi_graph_object_given_l import Full_Multi_Graph_Object_given_l
from src.common.rmp_graph_given_1 import RMP_graph_given_l
from src.common.pgm_approach import PGM_appraoch
from src.algorithm.update_states.standard_CVRP import CVRP_state_update_function
from src.algorithm.gwo_pricing_solver import GWOPricingSolver
from src.algorithm.update_states.general_states_update import General_state_update
from src.common.visulizer import Visulizer
from collections import defaultdict
from src.common.helper import Helper
import time
import random
from src.common.time_profile import TimeProfiler
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
        self.action_dict = actions
        self.actions = set().union(*actions.values())
        self.rhs_exog_vec = rhs_exog_vec
        self.initial_resource_state = initial_resource_state
        self.initial_resource_vector = initial_resource_vector
        self.initial_res_states = initial_res_states
        self.initial_res_actions = initial_res_actions
        self.state_update_function_cvrp = state_update_module
        #self.state_update_function = state_update_module[1]
        self.general_state_update = General_state_update(nodes, actions, initial_resource_vector,resource_name_to_index,number_of_resources)
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
        
        self.jy_options_user_defined=dict()
        self.jy_options_user_defined['epsilon']=1
        self.jy_options_user_defined['tolerance_compress']=100
        self.jy_options_user_defined['allow_compression']=True
        self.jy_options_user_defined['debug'] =True
        self.jy_options_user_defined['use_csr_exog'] =False
        self.gwo_pricing_solver = GWOPricingSolver(actions,initial_resource_state,nodes, self.resource_name_to_index,initial_resource_vector,self.jy_options_user_defined)
        random.seed(1000)
    def debug_check_duplicates(self,res_states):
        res_states_list=list(res_states)
        for i in range(0,len(res_states_list)):
            for j in range(0,len(res_states_list)):
                if i!=j:
                    if res_states_list[i].equals(res_states_list[j]):
                        # print('[i,j]')
                        # print([i,j])
                        res_states[i].pretty_print_state()
                        res_states[j].pretty_print_state()
                        input('error here')
    
    def solve(self):
        all_time_profile = defaultdict(float)
        all_time_start = time.time()
        with TimeProfiler(all_time_profile, "all_time"):
            
            l_id = 0
            max_iterations = 100000

            state_remove=[]
            for s in self.initial_res_states:
                
                if s.node==-2 and np.sum(s.state_vec)>0.5:
                    state_remove.append(s)
            for s in state_remove:

                self.initial_res_states.remove(s)

            my_init_graph=Full_Multi_Graph_Object_given_l(l_id, self.initial_res_states,self.actions, self.action_dict, self.dominate_actions,self.the_single_null_action,self.jy_options_user_defined)
            self.rez_states_minus:Set[State]=self.initial_res_states
            self.res_actions=self.initial_res_actions
            #l_id = 0
            #multi_graph = Full_Multi_Graph_Object_given_l(l_id,self.initial_res_states,self.initial_res_actions,self.dominate_actions)
            my_init_graph.initialize_system()
            all_time_profile = Helper.merge_two_dict(all_time_profile,my_init_graph.time_profile)

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
            self.complete_routes=[]
            print('starting Graph Master System')
            
            
            cg_iteration_time =1

            with TimeProfiler(all_time_profile, "solve:iteration"):
                while iteration < max_iterations:
                    time_profile = defaultdict(int)
                    #parameter for PGM

                    pgm_solver = PGM_appraoch(self.index_to_multi_graph,self.rhs_exog_vec, self.rez_states_minus,self.res_actions_minus,self.actions,incombentLP,self.dominate_actions,self.the_single_null_action,self.action_id_2_actions,self.lp_before_operations, self.jy_options_user_defined)
                    pgm_solver.call_PGM()
                    #this_visulizer = Visulizer(pgm_solver)
                    #this_visulizer.plot_graph()

                    pgm_solver.ilp_solve()
                    self.rez_states_minus, self.res_actions = pgm_solver.return_rez_states_minus_and_res_actions()
                    #all_time_profile['total_pgm_time'] += (pgm_time_end-pgm_time_start)
                    self.complete_routes=pgm_solver.complete_routes
                    all_time_profile = Helper.merge_two_dict(all_time_profile,pgm_solver.time_profile)

                    incombentLP = pgm_solver.cur_lp
                    self.lp_before_operations=pgm_solver.cur_lp

                    l_id += 1
                    #all action used in specific column 
                    states_used_in_this_col=set([])
                    new_states_describing_new_graph=[]
                    list_of_actions_used_in_col=set()
                    
                    if do_pricing==False:
                        #print('in pricing')
                        #print('in pricing')
                        beta_term, new_states_describing_new_graph,states_used_in_this_col = self.state_update_function_cvrp.get_states_from_random_beta(self.nodes, l_id)
                        reduced_cost = -np.inf
                    else:
                        #print('in not  pricing')
                        with TimeProfiler(all_time_profile, "solve:call_gwo_pricing"):
                        #[list_of_nodes_in_shortest_path, list_of_actions_used_in_col, reduced_cost]= self.pricing_problem.generalized_absolute_pricing(pgm_solver.dual_exog)
                            [list_of_nodes_in_shortest_path, list_of_actions_used_in_col, reduced_cost] = self.gwo_pricing_solver.call_gwo_pricing(pgm_solver.dual_exog)
                        with TimeProfiler(all_time_profile, "solve:get_new_states"):
                            trig = 0
                            # if trig==0:
                            max_depth, depth_used, states_used_in_this_col, node_min_vec_dict, action_reasonable, user_ignore_state_action,beta_info = self.state_update_function_cvrp._get_input(list_of_nodes_in_shortest_path,list_of_actions_used_in_col, l_id, self.initial_resource_state)
                            
                            new_states_describing_new_graph= self.general_state_update.state_generation(max_depth, depth_used, states_used_in_this_col, node_min_vec_dict, action_reasonable,user_ignore_state_action)
                            # elif trig==1:
                            #     beta_term, new_states_describing_new_graph,states_used_in_this_col=self.state_update_function_cvrp.get_new_states(list_of_nodes_in_shortest_path, list_of_actions_used_in_col,l_id)
                            # else:
                            #     max_depth, depth_used, states_used_in_this_col, min_vec_dict, action_reasonable, user_ignore_state_action, beta, beta_dict = self.state_update_function._get_input(list_of_nodes_in_shortest_path,list_of_actions_used_in_col, l_id)
                                
                            #     new_states_describing_new_graph= self.general_state_update.state_generation(max_depth, depth_used, states_used_in_this_col, min_vec_dict, action_reasonable, beta_dict,user_ignore_state_action)
                            #     beta_term, true_new_states_describing_new_graph,new_states_used_in_this_col=self.state_update_function_cvrp.get_new_states(list_of_nodes_in_shortest_path, list_of_actions_used_in_col,l_id,beta, beta_dict)

                            #     print('check true_new_states_describing_new_graph in new_states_describing_new_graph')
                            #     for s1 in true_new_states_describing_new_graph:
                            #         is_exsit = False
                            #         for s2 in new_states_describing_new_graph:
                            #             if s1.node == s2.node and np.array_equal(s1.state_vec.toarray(), s2.state_vec.toarray()):
                            #                 is_exsit = True
                            #                 break
                            #         if is_exsit == False:
                            #             print('error here')
                            #     print('check new_states_describing_new_graph in true_new_states_describing_new_graph')
                            #     for s1 in new_states_describing_new_graph:
                            #         is_exsit = False
                            #         for s2 in true_new_states_describing_new_graph:
                            #             if s1.node == s2.node and np.array_equal(s1.state_vec.toarray(), s2.state_vec.toarray()):
                            #                 is_exsit = True
                            #         if is_exsit == False:
                            #             print('error here')
                            #     print('finish check')
                        #debug
                        if self.jy_options_user_defined['debug'] == True:
                            with TimeProfiler(all_time_profile, "debug"):
                                for s1 in states_used_in_this_col:
                                    if s1 not in new_states_describing_new_graph:
                                        s1.pretty_print_state()
                                        input('error here this is not correct')
                        
                        print('shortest path reduce cost')
                        print(reduced_cost)
                    if reduced_cost >= -1e-5:
                        for index, graph in self.index_to_multi_graph.items():
                            all_time_profile = Helper.merge_two_dict(all_time_profile,graph.time_profile)
                        #all_time_profile['all_time'] = iteration_end_time- current_time
                        all_time_end = time.time()
                        all_time_profile['all_time'] = all_time_end - all_time_start
                        self.output_all_time_profile(all_time_profile)
                        return {
                            'status': 'optimal',
                            'x': pgm_solver.primal_sol,
                            'iterations': iteration,
                            'graph': self.index_to_multi_graph.values()
                        }
                    new_multi_graph = Full_Multi_Graph_Object_given_l(l_id,new_states_describing_new_graph,self.actions,self.action_dict,self.dominate_actions,self.the_single_null_action,self.jy_options_user_defined)


                    new_multi_graph.initialize_system()
                    #all_time_profile['multigraph total time'] += (multi_graph_end-multi_graph_start)

                    #all_time_profile = Helper.merge_two_dict(all_time_profile,new_multi_graph.time_profile)


                    self.index_to_multi_graph[l_id] = new_multi_graph
                    if self.jy_options_user_defined['debug']==True:
                        with TimeProfiler(all_time_profile, "debug"):
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
                                #input('julian predicts that these states will be the ones foudn to incduce errors')

                                for s in states_used_in_this_col:
                                    self.rez_states_minus.add(s)
                                    #s.pretty_print_state()
                                    if s not in new_multi_graph.rez_states:
                                        input('look this new state is not in the multigraph justadded ')
                                #debug here 
                                #input('-----')
                                self.debug_check_duplicates(self.rez_states_minus)
                    #all_time_profile['debug'] += (debug_end-debug_start)
                    iteration += 1
                    self.restricted_master_problem = 0
                    #input(' DONE A COMPLETE GM step')
                    print(f'========= cg iteration: {cg_iteration_time} =========')
                    cg_iteration_time+=1
                    #all_time_profile['all_time'] += time_spent
                    #self.output_all_time_profile(all_time_profile)
                return {'status': 'max_iterations', 'iterations': iteration}      
        
    

    
    def output_time_profile_for_this_iteration(self,time_profile):
        time_profile = time_profile
        time_sum = sum(sum(value.values()) for value in time_profile.values())
        output_key = ['pgm:ilp_solve_time','pgm:rmp_construct_time','pgm:lp_time', 'pgm:pricing_time','solve:pricing_time','multigraph:full_multigraph_construct_total_time']
        
        results = []
        for outer_key, inner_dict in time_profile.items():
            for inner_key, value in inner_dict.items():
                percentage = (value / time_sum) * 100
                results.append((outer_key, inner_key, value, percentage))
        results.sort(key=lambda x: x[2], reverse=True)
        this_percent = 0
        for outer_key, inner_key, value, percentage in results:
            if f'{outer_key} : {inner_key}' in output_key:
                print(f"{outer_key} : {inner_key} seconds {value:.4f} percent {percentage:.2f}%")
                this_percent += percentage
                if this_percent>99:
                    break
        print('stop output')
        return results
        # for key, value in self.time_profile.items():
        #     for step, duration in sorted(value.items(), key=lambda x: x[1], reverse=True):
        #         print(f"{step}: {duration:.4f} seconds ({duration/time_sum*100:.1f}%)")
    def output_all_time_profile(self, all_time_profile, output_file=None):
        """
        Outputs a comprehensive time profile for the GraphMaster.
        
        Args:
            all_time_profile: Combined dictionary of timing data from all iterations
            output_file (str, optional): If provided, output will be saved to this file
        """
        # Calculate total time
        total_time = all_time_profile.get('all_time', 0)
        # if total_time == 0:
        #     total_time = sum(time for op, time in all_time_profile.items() if op != 'all_time')
        
        # Print header
        print("\n=== Complete GraphMaster Time Profile ===")
        print(f"Total execution time: {total_time:.4f} seconds")
        print("-" * 50)
        
        # # Group by component type
        # component_times = {
        #     'solve': {'total': 0, 'operations': {}},
        #     'pgm': {'total': 0, 'operations': {}},
        #     'multi_graph': {'total': 0, 'operations': {}},
        #     'rmp': {'total': 0, 'operations': {}},
        #     'other': {'total': 0, 'operations': {}}
        # }
        
        # # Categorize operations
        # for operation, duration in all_time_profile.items():
        #     # Skip the 'all_time' entry which is the total
        #     if operation == 'all_time':
        #         continue
                
        #     # Determine component prefix
        #     if operation.startswith('solve:'):
        #         prefix = 'solve'
        #     elif operation.startswith('pgm:'):
        #         prefix = 'pgm'
        #     elif operation.startswith('multi_graph:'):
        #         prefix = 'multi_graph'
        #     elif operation.startswith('rmp:'):
        #         prefix = 'rmp'
        #     else:
        #         prefix = 'other'
            
        #     # Add to the appropriate component
        #     component_times[prefix]['total'] += duration
        #     component_times[prefix]['operations'][operation] = duration
        results = []
        for key, value in all_time_profile.items():
            if key != 'all_time':
                percentage = (value / total_time) * 100
                results.append((key,  value,percentage))
        results.sort(key=lambda x: x[1], reverse=True)
        for outer_key,  value, percentage in results:
            if percentage > 0.1:
                print(f"{outer_key}  seconds {value:.4f} percent {percentage:.2f}%")

        # Print hierarchical view
        # for prefix, data in sorted(component_times.items(), key=lambda x: x[1]['total'], reverse=True):
        #     prefix_total = data['total']
        #     percent = (prefix_total / total_time * 100) if total_time > 0 else 0
        #     print(f"{prefix} total: {prefix_total:.4f} seconds ({percent:.2f}%)")
            
        #     # Print operations within this component
        #     for op, dur in sorted(data['operations'].items(), key=lambda x: x[1], reverse=True):
        #         op_percent = (dur / total_time * 100) if total_time > 0 else 0
        #         if dur > 0.001:  # Skip very small durations
        #             print(f"  {op}: {dur:.4f} seconds ({op_percent:.2f}%)")
        #     print("-" * 30)
        
        # Write to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"=== Complete GraphMaster Time Profile ===\n")
                f.write(f"Total execution time: {total_time:.4f} seconds\n")
                f.write("-" * 50 + "\n")
                
                for operation, duration in sorted(all_time_profile.items(), key=lambda x: x[1], reverse=True):
                    if operation == 'all_time':
                        continue
                    percent = (duration / total_time * 100) if total_time > 0 else 0
                    f.write(f"{operation}: {duration:.4f} seconds ({percent:.2f}%)\n")
    def _solve_pricing(self, dual_vector: Dict[int, float]) -> Tuple[List[State], float]:
        """
        Calls the `pricer` solve method using the provided dual vector, and returns the path found.
        This might be redundant by itself. Perhaps replace this method with a `find_new_states` function that calls pricing and the state update function, all-in-one function.
        Also unsure why this function would return a Tuple[List[State], float], so that should probably be fixed.
        """
        # list_of_nodes, list_of_actions, total_cost = self.pricer.generalized_absolute_pricing(dual_vector)
        pass
    
