import numpy as np
from typing import Dict, Set, List, Tuple, Optional
from src.common.state import State
from src.common.action import Action
from src.algorithm.pricing_problem import PricingProblem
from src.algorithm.update_states.state_update_function import StateUpdateFunction
from src.common.full_multi_graph_object_given_l import Full_Multi_Graph_Object_given_l
from src.common.rmp_graph_given_1 import RMP_graph_given_l
from src.common.pgm_approach import PGM_appraoch
from src.common.pgm_approach import Route
from src.algorithm.update_states.standard_CVRP import CVRP_state_update_function
from src.algorithm.gwo_pricing_solver import GWOPricingSolver
from src.algorithm.gwo_pricing_solver_LoadAI import GWOPricingSolverLoadAI
from src.algorithm.update_states.general_states_update import General_state_update
from src.algorithm.update_states.jy_load_ai_state_generation import jy_make_load_ai_states
from src.common.visulizer import Visulizer
from collections import defaultdict
from src.common.helper import Helper
from src.algorithm.jy_slow_pricing import jy_slow_general_pricing_solver
from src.common.jy_fast_pricing import jy_fast_pricing
import time
import random
from src.algorithm.cg_rmp import CG_RMP
from src.algorithm.xz_jy_cg_solver_via import xy_jy_cg_solver
import random
import pickle
class GraphMaster_cg:
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
                 the_single_null_action: Action,
                 neighbors,
                 benefit_group,
                 benefit_group_cost
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
        self.state_update_module = state_update_module
        #self.state_update_function = state_update_module[1]
        self.general_state_update = General_state_update(nodes, actions, initial_resource_vector,resource_name_to_index,number_of_resources)

        self.dominate_actions = initial_dominate_actions
        self.resource_name_to_index = resource_name_to_index
        self.number_of_resources = number_of_resources
        self.the_single_null_action=the_single_null_action
        self.neighbors = neighbors
        self.benefit_group = benefit_group
        self.benefit_group_cost = benefit_group_cost
        #self.node_to_list = node_to_list
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
        self.jy_options_user_defined['use_load_ai_in_pgm'] =True
        self.jy_options_user_defined['using_load_ai_lazy']=True
        self.jy_options_user_defined['using_load_ai_lazy_num_pickups']=(len(nodes)-2)/3
        self.jy_options_user_defined['using_load_ai_lazy_max_pickups']=3
        self.jy_options_user_defined['max_actions_in_route']=len(nodes)+2
        self.jy_options_user_defined['max_pickups_in_a_route']=3
        self.jy_options_user_defined['use_cg'] = True
        self.jy_options_user_defined['complementary_col'] = 1
        self.jy_options_user_defined['use_fast_pricing'] = True
        self.jy_options_user_defined['lb_option'] =2
        self.jy_options_user_defined['poss_action'] = 2
        self.jy_options_user_defined['k_benefit_group'] = 10
        self.jy_options_user_defined['information_for_iteration'] =True
        self.jy_options_user_defined['use_comp_col'] =True # true: if use complementary column
        self.jy_options_user_defined['new_rmp'] =True # true: if generate more routes from omega term
        self.jy_options_user_defined['subset_route'] = True # true: if use subset routes for col service 3 customers
        self.jy_options_user_defined['optimality_gap'] = 0.01 
        self.jy_options_user_defined['min_dual_val_expand']=-1
        if self.jy_options_user_defined['use_load_ai_in_pgm']==True:
            self.jy_options_user_defined['max_actions_in_route']=2+(self.jy_options_user_defined['using_load_ai_lazy_max_pickups']*2)
            self.LOAD_AI_setup()
        #self.gwo_pricing_solver = GWOPricingSolver(actions,initial_resource_state,nodes, self.resource_name_to_index,initial_resource_vector,self.jy_options_user_defined)
        #self.gwo_pricing_solver_loadAI = GWOPricingSolverLoadAI(actions,initial_resource_state,nodes, self.resource_name_to_index,initial_resource_vector,self.jy_options_user_defined,self.state_update_module)
        random.seed(1000)
    def LOAD_AI_setup(self):
        load_ai_dict=dict()
        NC=(len(self.nodes)-2)/3
        NC=int(NC)
        load_ai_dict['num_pickups']=NC
        my_list_pickups=np.arange(1,NC+1).astype('int')
        pickup_nodes=set(my_list_pickups)
        load_ai_dict['pickup_nodes']=pickup_nodes
        my_list_dropoffs=np.arange(NC+1,(2*NC)+1).astype('int')
        print(my_list_dropoffs)
        drop_off_nodes=set(my_list_dropoffs)
        load_ai_dict['drop_off_nodes']=drop_off_nodes
        my_list_AUX=np.arange((NC*2)+1,(3*NC)+1).astype('int')

        aux_nodes=set(my_list_AUX)
        load_ai_dict['aux_nodes']=aux_nodes

        pickup_node_2_dropoff_node=dict()
        dropoff_node_2_pickup_node=dict()
        for i in pickup_nodes:
            pickup_node_2_dropoff_node[i]=i+load_ai_dict['num_pickups']
        load_ai_dict['pickup_node_2_dropoff_node']=pickup_node_2_dropoff_node

        for i in drop_off_nodes:
            dropoff_node_2_pickup_node[i]=i-load_ai_dict['num_pickups']
        load_ai_dict['dropoff_node_2_pickup_node']=dropoff_node_2_pickup_node

        print('self.resource_name_to_index')
        print(self.resource_name_to_index)
        node_2_may_pickup_resource_number=dict()
        node_2_must_dropoff_resource_number=dict()
        for i in pickup_nodes:
            j=i+load_ai_dict['num_pickups']
            my_name_may_pickup=str(("may_pickup", i))
            my_name_may_avoid_dropoff=str(("may_avoid_dropoff", j))
            idx_pickup=self.resource_name_to_index[my_name_may_pickup]
            idx_dropoff=self.resource_name_to_index[my_name_may_avoid_dropoff]
            node_2_may_pickup_resource_number[i]=idx_pickup
            node_2_may_pickup_resource_number[j]=idx_pickup
            node_2_must_dropoff_resource_number[i]=idx_dropoff
            node_2_must_dropoff_resource_number[j]=idx_dropoff
            #self.resource_name_to_index[my_name_pickup]
        load_ai_dict['node_2_may_pickup_resource_number']=node_2_may_pickup_resource_number
        load_ai_dict['node_2_must_dropoff_resource_number']=node_2_must_dropoff_resource_number
        self.jy_options_user_defined['load_ai_dict']=load_ai_dict
        print('hi')

    def LOAD_AI_CHECK_Valid_states(self):
        #pn, dn for a pickup and dropoff pair
        #kills so anything that says that dropoff 
        #possibilities (may pick up )
        
        print('fill me in ')
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
    def _initial_routes(self):
        actions_from_minus1 = {}  # Actions starting at node -1
        actions_to_minus2 = {}  
        for action in self.res_actions_minus:
            if action.node_tail == -1:
                # Store actions starting from -1 indexed by their destination
                actions_from_minus1[action.node_head] = action
            if action.node_head == -2:
                # Store actions ending at -2 indexed by their origin
                actions_to_minus2[action.node_tail] = action
        
        # Find intermediate nodes that appear in both dictionaries
        common_nodes = set(actions_from_minus1.keys()).intersection(set(actions_to_minus2.keys()))
        list_of_routes = []
        for node in common_nodes:
            source_state = State(-1,self.initial_resource_vector,0,True,False)
            state_action = [source_state]
            first_action = actions_from_minus1[node]
            state_action.append(first_action)
            this_state = first_action.get_head_state_fast_load_ai(source_state,source_state.l_id)
            state_action.append(this_state)
            second_action = actions_to_minus2[node]
            state_action.append(second_action)
            this_state = second_action.get_head_state_fast_load_ai(this_state,source_state.l_id)
            state_action.append(this_state)
            route = Route(state_action,1,self.state_update_module.pickup_node)
            list_of_routes.append(route)
        return list_of_routes
    
    def _check_path_duplicate(self,path,red_cost):
        path_tuple = tuple(path)
        if path_tuple in self.path_added and red_cost<-0.00001:
            input(f'error: path{path_tuple} added before with reduce cost: {red_cost}')
        else:
            self.path_added.add(path_tuple)
    def solve(self):
        all_time_profile = defaultdict(float)
        all_time_start = time.time()
        jy_actions_node=defaultdict(list)
        for (n1,n2),a_list in self.action_dict.items():
            jy_actions_node[n1].append(a_list[0])

            
        l_id = 0
        max_iterations = 100000
        state_remove=[]
        for s in self.initial_res_states:
            if s.node==-2 and np.sum(s.state_vec)>0.5:
                state_remove.append(s)
        for s in state_remove:
            self.initial_res_states.remove(s)
        self.rez_states_minus:Set[State]=self.initial_res_states
        self.res_actions=self.initial_res_actions
        iteration = 1
        self.action_id_2_actions={my_action.action_id: my_action for my_action in self.actions}
        self.lp_before_operations=np.inf
        self.complete_routes=[]
        print('starting Graph Master System')
        
        cg_iteration_time =1
        self.path_added = set()
        list_of_routes = self._initial_routes()
        forbidden_omega = []
        node_sequence_of_routes = []
        all_reduce_cost_list = []
        all_min_reduce_cost = []
        num_path_col_in_rmp = []
        num_omega_col_in_rmp = []
        path_col_generated = []
        path_added = []
        for route in list_of_routes:
            node_sequence_of_routes.append(route.node_in_ordered)

        lp_objective_list = []
        omega_term_list = []
        cg_solver = xy_jy_cg_solver(list_of_routes,node_sequence_of_routes,self.rhs_exog_vec,self.state_update_module,forbidden_omega,self.initial_resource_vector)
        while iteration < max_iterations:
            print(type(self.state_update_module.actions))
            
            if self.jy_options_user_defined['new_rmp'] == True:
                
                #output = cg_solver.solve()
                #input('before')
                
                before_num = len(list_of_routes)
                output,list_of_routes, node_sequence_of_routes = cg_solver.solve_2()
                print('before : len(list_of_routes)')
                print(before_num)
                print('after : len(list_of_routes)')
                print(len(list_of_routes))
                print('check')
                #input('during')
                #output = cg_solver.solve()
                #input('after')

            else:
                output = cg_solver.solve()
            all_time_profile = Helper.merge_two_dict(all_time_profile,cg_solver.time_profile)
            num_path_col_in_rmp.append(cg_solver.col_of_path)
            num_omega_col_in_rmp.append(cg_solver.col_of_omega)
            this_sol = output['variable_values']
            this_dual = output['dual_values']
            this_lp_objective = output['objective_value']
            lp_objective_list.append(this_lp_objective)
            for index, route_index in cg_solver.var_index_to_route_index.items():
                value = this_sol[index]
                if value > 0.0001:
                    this_route = list_of_routes[route_index]
                    red_cost = this_route.get_red_cost(this_dual)
                    if red_cost <-1:
                        input('error here')
                    print(f'value:{value}, red_cost:{red_cost}')
            l_id += 1

            jy_init_res_state = State(-1,self.initial_resource_vector,l_id,True,False)
            this_dual = [0 if abs(x) < 0.0001 else x for x in this_dual]
            if self.jy_options_user_defined['use_fast_pricing'] == True:
               
                jy_fast_pricer = jy_fast_pricing(self.actions,self.action_dict,this_dual,jy_init_res_state,self.jy_options_user_defined['max_actions_in_route'],jy_actions_node,self.nodes,self.neighbors, self.benefit_group,self.benefit_group_cost,self.jy_options_user_defined)
                routes= jy_fast_pricer.run()
                reduced_cost_list = [r.get_red_cost(this_dual) for r in routes]
                all_reduce_cost_list.append(reduced_cost_list)
                reduced_cost=0
                if len(reduced_cost_list)>0:
                    reduced_cost = min(reduced_cost_list)
                    all_min_reduce_cost.append(reduced_cost)
                print('reduced_cost')
                print(reduced_cost)
                print('reduced_cost_list')
                print(reduced_cost_list)
                #input('----')
                rmp_obj = output['objective_value']
                if reduced_cost >= -1.1 or abs(sum(x for x in reduced_cost_list if x < 0)) < rmp_obj*self.jy_options_user_defined['optimality_gap']:
                    this_forbidden_omega = cg_solver.get_active_DOI()
                    print('this_forbidden_omega')
                    print(this_forbidden_omega)
                    omega_term_list.append(len(this_forbidden_omega))
                    output_info = defaultdict()
                    if len(this_forbidden_omega)<0.5:
                        if self.jy_options_user_defined['information_for_iteration'] == True:
                            output_info['list of lp'] = lp_objective_list
                            output_info['list number of positive omega terms (for each iteration lp'] = omega_term_list
                            output_info['the sum of the omega terms (all positive omega)'] = forbidden_omega
                            output_info['the total number of forbidden omega terms'] = len(forbidden_omega)
                            output_info['sum of the reduced cost term (reduce cost of all path each iteration)'] = all_reduce_cost_list
                            output_info['minimum reduced cost term (for each iteration)'] = all_min_reduce_cost
                            output_info['number of col (path) in rmp (for each iteraiton)'] = num_path_col_in_rmp
                            output_info['number of col (omega) in rmp (for each iteraiton)'] = num_omega_col_in_rmp
                            output_info['number of col generated (path added for each iteration)'] = path_col_generated
                            
                        ilp_cg_solver = xy_jy_cg_solver(list_of_routes,node_sequence_of_routes,self.rhs_exog_vec,self.state_update_module,forbidden_omega,self.initial_resource_vector)
                        
                        sol = ilp_cg_solver.solve_ilp()
                        last_lp_obj = rmp_obj
                        ilp_obj = sol['objective_value']
                        gap = (ilp_obj-last_lp_obj)/last_lp_obj
                        all_time_profile = Helper.merge_two_dict(all_time_profile,ilp_cg_solver.time_profile)
                        all_time_end = time.time()
                        all_time_profile['all_time'] = all_time_end - all_time_start
                        
                        used_routes = sol['used_routes']
                        #variable_to_value = sol['variable_values']
                        print('route generated')
                        for route in list_of_routes:
                            print(route.node_in_ordered)
                        print('route used')
                        
                        for route in used_routes:
                            print(route.node_in_ordered)
                            valid = self.validate_route(route)
                            if valid == False:
                                input('invalid route here')
                        route_num = 1
                        for route in used_routes:
                            print(f'=========route {route_num}============')
                            print('node in route ordered')
                            print(route.node_in_ordered)
                            print('time remaining')
                            print([s.state_vec[:2] for s in route.just_states_ordered])
                            # if len(route.just_states_ordered) >3:
                            #     print('time window start')
                            #     print([self.state_update_module.time_window_start[s.node] for s in route.just_states_ordered])
                            #     print('time window end')
                            #     print([self.state_update_module.time_window_end[s.node] for s in route.just_states_ordered])
                            #     print('weight remain')
                            print([s.state_vec[0] for s in route.just_states_ordered])
                            print('volume remain')
                            print([s.state_vec[1] for s in route.just_states_ordered])
                            route_num+=1
                        print('=========time profiling================')
                        self.output_all_time_profile(all_time_profile)
                        with open("time.pkl", "wb") as f:
                            pickle.dump(all_time_profile, f)
                        return {
                            'status': 'optimal',
                            'x': sol['variable_values'],
                            'iterations': iteration,
                            'used_routes':used_routes,
                            'output_info':output_info,
                            'optimality_gap':gap
                        }
                    else:
                        forbidden_omega.extend(this_forbidden_omega)

                #list_of_routes.extend(routes)
                add_route_num = 0
                for idx in range(len(routes)):
                    route = routes[idx]
                    #red_cost = route.get_red_cost(this_dual)
                    red_cost = reduced_cost_list[idx]
                    if route.node_in_ordered in node_sequence_of_routes and red_cost<-1:
                        print('node_in_ordered')
                        print(route.node_in_ordered)
                        print('red_cost')
                        print(red_cost)
                        input('error here: route added has negative red cost')
                    if red_cost<-1e-3 :
                        print('route added')
                        print(route.node_in_ordered)
                        node_sequence_of_routes.append(route.node_in_ordered)
                        list_of_routes.append(route)
                        cg_solver.add_route(route)
                        if self.jy_options_user_defined['subset_route'] == True:
                            if len(route.node_in_ordered)>=2+self.jy_options_user_defined['max_pickups_in_a_route']*2:
                                subset_of_routes = route.generate_subset_routes()
                                for subset_route in subset_of_routes:
                                    if subset_route not in node_sequence_of_routes:
                                        cur_state = State(-1,self.initial_resource_vector,1,True,False)
                                        state_action_alt_repeat=[cur_state]
                                        for o,d in zip(subset_route[:-1],subset_route[1:]):
                                            this_a:Action = self.action_dict[(o,d)][0]
                                            state_action_alt_repeat.append(this_a)
                                            try:
                                                new_state = this_a.get_head_state_fast_load_ai(cur_state,1)
                                            except:
                                                print('check here')
                                            state_action_alt_repeat.append(new_state)
                                            cur_state = new_state
                                        
                                        this_sub_route = Route(state_action_alt_repeat,1,self.state_update_module.pickup_node)
                                        list_of_routes.append(this_sub_route)
                                        node_sequence_of_routes.append(this_sub_route.node_in_ordered)
                        add_route_num += 1
                path_col_generated.append(add_route_num)
                
                print('======route check here======')
            else:
                for i in range(self.jy_options_user_defined['complementary_col']):
                    jy_pricer_my =jy_slow_general_pricing_solver(self.actions,this_dual,jy_init_res_state,self.jy_options_user_defined['max_actions_in_route'],jy_actions_node,self.nodes,self.jy_options_user_defined)
                    [list_of_nodes_in_shortest_path, list_of_actions_used_in_col, state_in_ordered,reduced_cost,jy_actions_node] =jy_pricer_my.return_solution()
                    print('done jy pricing ')
                    state_action_list =[]
                    for idx in range(len(list_of_actions_used_in_col)):
                        state_action_list.append(state_in_ordered[idx])
                        state_action_list.append(list_of_actions_used_in_col[idx])
                    state_action_list.append(state_in_ordered[-1])
                    this_route = Route(state_action_list,1,self.state_update_module.pickup_node)
                    list_of_routes.append(this_route)
                    self._check_path_duplicate(list_of_nodes_in_shortest_path,reduced_cost)
                    nonzero_indices = np.nonzero(this_route.Exog_vec)[0]
                    for idx in nonzero_indices:
                        this_dual[idx] =0
                
            
            iteration += 1
            print(f'========= cg iteration: {cg_iteration_time} =========')
            cg_iteration_time+=1
        return {'status': 'max_iterations', 'iterations': iteration}     
    def validate_route(self,route):
        return route.verify_feasibility()
    def post_procssing(self, routes):
        routes_no_over_cover = routes[:]
        rhs_sum = np.zeros(len(self.rhs_exog_vec))
        node_to_routes = defaultdict(list)
        for route in routes:
            rhs_sum += route.Exog_vec
            non_zero_indices = np.nonzero(route.Exog_vec)[0]
            for node in non_zero_indices:
                node_to_routes[node+1].append(route)
        over_cover = rhs_sum - np.ones(self.rhs_exog_vec)
        over_cover_indices = np.nonzero(over_cover)[0]
        for idx in over_cover_indices:
            over_cover_num = over_cover_indices[idx]
            over_cover_node = idx+1
            random_route_remove = random.sample(node_to_routes[over_cover_node],over_cover_num)
            
            for route in random_route_remove:
                state_action_alt_repeat = []
                node_in_ordered = route.node_in_ordered
                node_in_ordered.remove(over_cover_node)
                node_in_ordered.remove(over_cover_node+len(self.pickup_node))
                cur_state = State(-1,self.initial_resource_vector,0,True,False)
                for (tail,head) in zip(node_in_ordered[:-1],node_in_ordered[1:]):
                    this_act = self.actions[(tail,head)][0]
                    state_action_alt_repeat.append(this_act)
                    next_state = this_act.get_head_state(cur_state)
                    if next_state == None:
                        input('error here: none state generated from given column')
                    state_action_alt_repeat.append(next_state)
                    cur_state = next_state
                this_route = Route(state_action_alt_repeat,1,self.state_update_module.pickup_node)
                routes_no_over_cover.remove(route)
                routes_no_over_cover.append(this_route)
    

    
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
    
