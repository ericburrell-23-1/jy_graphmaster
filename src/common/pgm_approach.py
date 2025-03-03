from src.common.rmp_graph_given_1 import RMP_graph_given_l
from src.common.full_multi_graph_object_given_l import Full_Multi_Graph_Object_given_l
from src.common.action import Action
from src.common.state import State
from collections import defaultdict
from src.common.jy_var import jy_var
#from src.common.route import route
from typing import Dict, DefaultDict, Set, List
import numpy as np
import pulp as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xpress as xp
import networkx as nx

class route:
    def __init__(self,state_action_alt_repeat,weight):
        self.state_action_alt_repeat=state_action_alt_repeat
        self.weight=weight

class PGM_appraoch:
    #This will do the enitre RMP. 

    #This will do the enitre RMP. 


    def __init__(self,index_to_graph:dict[Full_Multi_Graph_Object_given_l],prob_RHS,rez_states_minus: Set[State],rez_actions_minus,incumbant_lp,dominated_action,the_null_action,action_id_2_actions,lp_before_operations):
        self.index_to_graph:DefaultDict[int,Full_Multi_Graph_Object_given_l] = index_to_graph
        self.my_PGM_graph_list:List[Full_Multi_Graph_Object_given_l]=list(self.index_to_graph.values()) #list of all of the PGM graphs
        self.prob_RHS:np.ndarray=prob_RHS #RHS
        self.action_id_2_actions=action_id_2_actions
        self.action_id_2_actions=action_id_2_actions
        self.rez_states_minus:Set[State]=rez_states_minus #has all of the states in rez states minus by graph . So if I put in a grpah id then i get out the rez states minus to initialize
        self.rez_actions_minus=rez_actions_minus #get all actions that are currently under consdieration
        self.incumbant_lp=incumbant_lp# has incumbent LP objective
        self.the_null_action = the_null_action
        self.dominated_actions = dominated_action
        self.init_defualt_jy_options()
        self.lp_before_operations=lp_before_operations
        self.make_rez_states_minus_by_node()

        self.primal_solution, self.dual_solution, self.optimal_value = None, None, None

    #put your stuff here start with marting debug

    #i want to take in two lists of states.  I want to determine if one list is a subset of the other list
    

    def debug_check_elem_res_nodes(self):
        for g_id in self.rez_states_minus_by_node:
            for n in self.rez_states_minus_by_node[g_id]:
                orig_len=len(self.rez_states_minus_by_node[g_id][n])
                new_len=len(set(self.rez_states_minus_by_node[g_id][n]))
                if orig_len!=new_len:
                    print('[g_id,n,orig_len,new_len]')
                    print([g_id,n,orig_len,new_len])
                    print('self.rez_states_minus_by_node')
                    print(self.rez_states_minus_by_node[g_id][n])
                    input('error here ')
    def is_state_set_subset(self,original:List[State],subset:List[State]):
        flag = True
        for state in subset:
            found  = False
            for state2 in original:
                found = found or state.equals(state2)
            flag = flag and found
        return flag
    
    def compare_rez_states(self,rez_states_by_node_1,rez_states_by_node_2):
        keys1 = set(rez_states_by_node_1.keys())
        keys2 = set(rez_states_by_node_2.keys())
        if not keys2 == keys1:
            return False
        
        for l_id  in keys1:
            nodes_1 = set(rez_states_by_node_1[l_id])
            nodes_2 = set(rez_states_by_node_2[l_id])

            if not nodes_2==nodes_1:
                return False
            
            flag = True
            for node in nodes_1:
                flag =flag and self.is_state_set_subset(rez_states_by_node_1[l_id][node],rez_states_by_node_2[l_id][node])

        return flag
 

    #this has to work though and we have had issues with the comparison operator
 
    #second.  It would be useful if we could take in a res_states_by_node which is a dictionary of [graph_id][node_id] followed by a set and determine if res_states_by_node_1 is a subset of res_states_by_node_2 for each element 
 

    def decoder_make_content_info(self,primal_solution):
        self.decoder_gid_2_fill=dict()
        epsilon=.00001
        for my_var in primal_solution:
            if primal_solution[my_var]>epsilon:
                if my_var[0]=='eq_act_var':
                    g_id=my_var[1]
                    eqiv_class=my_var[2]
                    action_id=my_var[3]
                    if g_id not in self.decoder_gid_2_fill:
                        self.decoder_gid_2_fill[g_id]=dict()
                    if eqiv_class not in self.decoder_gid_2_fill[g_id]:
                        self.decoder_gid_2_fill[g_id][eqiv_class]=dict()
                    self.decoder_gid_2_fill[g_id][eqiv_class][action_id]=primal_solution[my_var]
                    if len(self.decoder_gid_2_fill[g_id][eqiv_class])>1:
                        print('self.decoder_gid_2_fill[g_id][eqiv_class][action_id]')
                        print(self.decoder_gid_2_fill[g_id][eqiv_class][action_id])
                        input('in principle this is ok but not in the CVRP stuf that I am doing ')              
    def decoder_make_graphs(self,primal_solution):
        epsilon=.00001
        print('----')
        self.decoder_gid_2_edges=dict()

        for g_id in self.index_to_graph:
            self.decoder_gid_2_edges[g_id]=dict()

        for my_var in primal_solution:
            if primal_solution[my_var]>epsilon:
                if my_var[0]=='edge':
                    s1_id=my_var[2]
                    s2_id=my_var[3]
                    g_id=my_var[1]
                    g=self.index_to_graph[g_id]
                    gr=self.pgm_graph_2_rmp_graph[g]
                    s1=gr.state_id_to_state[s1_id]
                    s2=gr.state_id_to_state[s2_id]
                    my_equiv_class=gr.s1_s2_pair_2_equiv[(s1,s2)]
                    x_val=primal_solution[my_var]#.copy()
                    
                    self.decoder_gid_2_edges[g_id][(s1_id,s2_id)]=x_val#primal_solution[my_var].copy()#,s1_id,s2_id])  
    def decoder_iter_make_paths(self,my_edges_orig,source_state_id,sink_state_id):
        
        epsilon=.00001
        my_edges=my_edges_orig.copy()
        
        my_paths=[]
        epsilon=.000001
        while(True):
            G = nx.DiGraph()
            tot_weight=0
            for (s1_id,s2_id) in my_edges:
                this_weight=my_edges[(s1_id,s2_id)]
                if this_weight>epsilon:
                    G.add_edge(s1_id,s2_id,weight=this_weight)
                    tot_weight=tot_weight+this_weight
            if tot_weight<epsilon*10:
                break
            shortest_path = nx.bellman_ford_path(G, source=source_state_id, target=sink_state_id, weight="weight")
            shortest_path_length = nx.bellman_ford_path_length(G, source=source_state_id, target=sink_state_id, weight='weight')
            
            new_path=[]
            for i in range(0,len(shortest_path)):
                new_path.append(shortest_path[i])

            amount_subtract=np.inf
            for i in range(0,len(shortest_path)-1):
                tail=shortest_path[i]
                head=shortest_path[i+1]
                my_weight=my_edges[(tail,head)]

                if my_weight<amount_subtract:
                    amount_subtract=my_weight
            new_path.append(amount_subtract)
            for i in range(0,len(shortest_path)-1):
                tail=shortest_path[i]
                head=shortest_path[i+1]
                my_edges[(tail,head)]=my_edges[(tail,head)]-amount_subtract
                if my_edges[(tail,head)]<-epsilon:
                    print('my_edges_orig[(tail,head)]')
                    print(my_edges_orig[(tail,head)])
                    
                    input('error here')
            my_paths.append(new_path) 
        return my_paths
    def decode_sol_2_paths(self,primal_sol):
        self.decoder_make_graphs(primal_sol)
        self.decoder_make_content_info(primal_sol)
        self.my_paths_no_action_by_g_id=dict()
        for g_id in self.index_to_graph:
            self.my_paths_no_action_by_g_id[g_id]=dict()
            if len(self.decoder_gid_2_edges[g_id])>0:
                source_state_id=self.index_to_graph[g_id].source_state.state_id
                sink_state_id=self.index_to_graph[g_id].sink_state.state_id
                my_paths=self.decoder_iter_make_paths(self.decoder_gid_2_edges[g_id],source_state_id,sink_state_id)
                self.my_paths_no_action_by_g_id[g_id]=my_paths
        #self.decoder_fill_in_content()
        self.complete_routes=[]
        for g_id in self.index_to_graph:
            if len(self.decoder_gid_2_edges[g_id])>0:
                
                for p_info in self.my_paths_no_action_by_g_id[g_id]:

                    self.decoder_create_route_info(g_id,p_info)
                    #for my_new_route in some_new_routes:
                        #my_new_route=self.decoder_create_actual_route_from_data(g_id,my_new_route_info)
                   #     self.complete_routes.add(my_new_route)
    def decoder_create_route_info(self,g_id,p_info):
        epsilon=.000001
        tot_weight_rem=p_info[-1]
        state_id_on_route=p_info[0:-1]
        state_first_on_route=self.index_to_graph[g_id].source_state.state_id#state_id_on_route[0]
        g=self.index_to_graph[g_id]
        s1s2_2_equiv=self.pgm_graph_2_rmp_graph[g].s1_s2_pair_2_equiv
        path_length=len(state_id_on_route)
        s1_id_s2_id_2_equiv=dict()
        for i in range(0,path_length-1):
            s1_id=state_id_on_route[i]
            s2_id=state_id_on_route[i+1]
            s1=self.pgm_graph_2_rmp_graph[g].state_id_to_state[s1_id]
            s2=self.pgm_graph_2_rmp_graph[g].state_id_to_state[s2_id]
            my_equiv=s1s2_2_equiv[(s1,s2)]
            s1_id_s2_id_2_equiv[(s1_id,s2_id)]=my_equiv
        while tot_weight_rem>0:
            this_path_s1_act_s2_repeat=[state_first_on_route]
            min_val_in_path=tot_weight_rem
            s1_id_s2_id_2_act_id_on_path=dict()
            for i in range(0,path_length-1):
                s1_id=state_id_on_route[i]
                s2_id=state_id_on_route[i+1]
                my_equiv=s1_id_s2_id_2_equiv[(s1_id,s2_id)]
                did_find=False
                for my_act_id in self.decoder_gid_2_fill[g_id][my_equiv]:
                    this_weight=self.decoder_gid_2_fill[g_id][my_equiv][my_act_id]
                    if this_weight>epsilon:
                        min_val_in_path=min([this_weight,min_val_in_path])
                        did_find=True
                        my_act=self.action_id_2_actions[my_act_id]
                        this_path_s1_act_s2_repeat.append(my_act)
                        this_path_s1_act_s2_repeat.append(s2)
                        s1_id_s2_id_2_act_id_on_path[(s1_id,s2_id)]=my_act_id
                        break
                if did_find==False:
                    print('self.decoder_gid_2_fill[g_id][my_equiv]')
                    print(self.decoder_gid_2_fill[g_id][my_equiv])
                    input('error here not able to fill')
            for i in range(0,path_length-1):
                s1_id=state_id_on_route[i]
                s2_id=state_id_on_route[i+1]
                my_equiv=s1_id_s2_id_2_equiv[(s1_id,s2_id)]
                my_act_id=s1_id_s2_id_2_act_id_on_path[(s1_id,s2_id)]
                self.decoder_gid_2_fill[g_id][my_equiv][my_act_id]-=min_val_in_path
                if self.decoder_gid_2_fill[g_id][my_equiv][my_act_id]<-epsilon:
                    input('error here')
            tot_weight_rem=tot_weight_rem-min_val_in_path
            my_new_route=route(this_path_s1_act_s2_repeat,min_val_in_path)
            self.complete_routes.append(my_new_route)
        print('create route info done')
    def debug_check_primal_exog_feas(self,primal_solution):
        contrib_RHS_tot=self.prob_RHS*0
        for my_var in primal_solution:
            if primal_solution[my_var]>0:
                if my_var[0]=='eq_act_var':
                    my_action_id=my_var[3]
                    my_action=self.action_id_2_actions[my_action_id]
                    contrib_RHS_tot+=primal_solution[my_var]*my_action.Exog_vec
        if np.sum(np.abs(contrib_RHS_tot-self.prob_RHS))>.0001:
            print('contrib_RHS_tot')
            print(contrib_RHS_tot)
            print('self.prob_RHS')
            print(self.prob_RHS)
            input('error in sum')
    def  debug_check_primal_solution_match(self,primal_solution):

        equiv_class_2_edge_count=dict()
        equiv_class_2_fill_count=dict()
        #fill equiv class 2 fill
        my_vals_by_graph=dict()
        tot_val_edge=0
        tot_val_fill=0
        epsilon=.000001

        #input('starting analysis solution')
        for g_id in self.index_to_graph:
            equiv_class_2_edge_count[g_id]=dict()
            equiv_class_2_fill_count[g_id]=dict()
        for my_var in primal_solution:
            if primal_solution[my_var]>epsilon:
                if my_var[0]=='eq_act_var':
                    g_id=my_var[1]
                    my_equiv_class=my_var[2]
                    if my_equiv_class not in equiv_class_2_fill_count[g_id]:
                        equiv_class_2_fill_count[g_id][my_equiv_class]=0
                    equiv_class_2_fill_count[g_id][my_equiv_class]+=primal_solution[my_var]
                    tot_val_fill+=primal_solution[my_var]
                    #print('var')
                    #print('value = '+str(primal_solution[my_var]))
                    #self.help_print_var(my_var)
        self.states_used_sol=set([])
        for my_var in primal_solution:
            if primal_solution[my_var]>epsilon:
                if my_var[0]=='edge':
                    s1_id=my_var[2]
                    s2_id=my_var[3]
                    
                    g_id=my_var[1]
                    g=self.index_to_graph[g_id]
                    gr=self.pgm_graph_2_rmp_graph[g]
                    s1=gr.state_id_to_state[s1_id]
                    s2=gr.state_id_to_state[s2_id]
                    self.states_used_sol.add(s1)
                    self.states_used_sol.add(s2)
                    my_equiv_class=gr.s1_s2_pair_2_equiv[(s1,s2)]
                    if my_equiv_class not in equiv_class_2_edge_count[g_id]:
                        equiv_class_2_edge_count[g_id][my_equiv_class]=0
                    equiv_class_2_edge_count[g_id][my_equiv_class]+=primal_solution[my_var]
                    tot_val_edge+=primal_solution[my_var]
                    print('var')
                    print('value = '+str(primal_solution[my_var]))
                    self.help_print_var(my_var)

        for g_id in self.index_to_graph:
            for my_equiv_class in equiv_class_2_edge_count[g_id]:
                if my_equiv_class not in equiv_class_2_fill_count[g_id]:
                    print('paused')
                    print()
                    print('equiv_class_2_fill_count[g_id]')
                    print(equiv_class_2_fill_count[g_id])
                    input('err here ')
                if 0.0001<np.abs(equiv_class_2_edge_count[g_id][my_equiv_class]-equiv_class_2_fill_count[g_id][my_equiv_class]):
                    print('equiv_class_2_edge_count[g_id][my_equiv_class]')
                    print(equiv_class_2_edge_count[g_id][my_equiv_class])
                    print('equiv_class_2_fill_count[g_id][my_equiv_class]')
                    print(equiv_class_2_fill_count[g_id][my_equiv_class])
                    print('g_id')
                    print(g_id)
                    print('my_equiv_class')
                    print(my_equiv_class)
                    input('error here 1')
        
        print('-----')
        print('-----')
        print('-----')
        print('-----')
        
        for g_id in self.index_to_graph:
            for my_equiv_class in equiv_class_2_fill_count[g_id]:
                #print('g_id')
                #print(g_id)
                #print('my_equiv_class')
                #print(my_equiv_class)
                if my_equiv_class not in equiv_class_2_edge_count[g_id]:
                    print('paused')
                    input('err here ')
                if 0.0001<np.abs(equiv_class_2_edge_count[g_id][my_equiv_class]-equiv_class_2_fill_count[g_id][my_equiv_class]):
                    print('equiv_class_2_edge_count[g_id][my_equiv_class]')
                    print(equiv_class_2_edge_count[g_id][my_equiv_class])
                    print('equiv_class_2_fill_count[g_id][my_equiv_class]')
                    print(equiv_class_2_fill_count[g_id][my_equiv_class])
                    print('g_id')
                    print(g_id)
                    print('my_equiv_class')
                    print(my_equiv_class)
                    input('error here 2')
        
        #print('equiv_class_2_fill_count')
        #print(equiv_class_2_fill_count)
        #print('equiv_class_2_edge_count')
        #print(equiv_class_2_edge_count)
        
        #print('tot_val_edge')
        #print(tot_val_edge)
        #print('tot_val_fill')
        #print(tot_val_fill)
        #input('paused here')
    def help_print_var(self,my_jy_var_name):

        #                    my_name = ('edge', g.l_id, s1.state_id, s2.state_id)
        #my_name = ('eq_act_var', g.l_id, my_eq, my_act.action_id)
        print('----')

        if my_jy_var_name[0]=='edge':
            print('edge')
            l_id=my_jy_var_name[1]
            s1_id=my_jy_var_name[2]
            s2_id=my_jy_var_name[3]

            print('g.l_id: '+str(l_id))
            s1=self.my_PGM_graph_list[l_id].state_id_to_state[s1_id]
            s2=self.my_PGM_graph_list[l_id].state_id_to_state[s2_id]
            s1.pretty_print_state()
            s2.pretty_print_state()

        else:
            print('eq_act_var')
            l_id=my_jy_var_name[1]
            my_eq=my_jy_var_name[2]
            my_action_id=my_jy_var_name[3]
            my_action=self.action_id_2_actions[my_action_id]
            print('g.l_id: '+str(l_id))
            print('my_eq: '+str(my_eq))
            my_action.pretty_print_action()

        print('*******')
    def make_rez_states_minus_by_node(self):
        """Groups states by (l_id, node) into a dictionary of lists with structure {l_id: {node: [states]}}."""

        self.rez_states_minus_by_node = defaultdict(lambda: defaultdict(set))  # Nested defaultdict for automatic list initialization
        self.rez_states_minus_by_graph: Dict[int, Set[State]] = defaultdict(set)
        for my_state in self.rez_states_minus:
            self.rez_states_minus_by_node[my_state.l_id][my_state.node].add(my_state)
            self.rez_states_minus_by_graph[my_state.l_id].add(my_state)
        # Check that each l_id has exactly one source and one sink
        for l_id in self.rez_states_minus_by_node:
            source_count = len(self.rez_states_minus_by_node[l_id].get(-1, []))
            sink_count = len(self.rez_states_minus_by_node[l_id].get(-2, []))

            if source_count != 1 or sink_count != 1:
                raise ValueError(f"Graph {l_id} must have exactly one source and one sink, but found {source_count} source(s) and {sink_count} sink(s).")

        #print('hello moose')
        #print('hello moose2')
        #print(self.rezStates_minus_by_node.keys())
        #input('moo')
    def init_defualt_jy_options(self):

        self.jy_options=dict()
        self.jy_options['epsilon']=.00001
        self.jy_options['tolerance_compress']=.00001
        self.jy_options['allow_compression']=True
    def put_all_nodes_actions_in_consideration_set(self):
        
        self.rez_actions_minus=set()
        for my_action_id in self.action_id_2_actions:
            my_action=self.action_id_2_actions[my_action_id]
            self.rez_actions_minus.add(my_action)

        self.rez_states_minus=set()
        for l_id in self.index_to_graph:#.items():
            g=self.index_to_graph[l_id]
            for s in g.rez_states:
                self.rez_states_minus.add(s)
        self.make_rez_states_minus_by_node()
    def call_GM_NO_PGM(self):

        self.put_all_nodes_actions_in_consideration_set()
        [self.primal_sol,self.dual_exog,self.cur_lp]=self.call_PGM_RMP_solver_from_scratch()#we can do better a different time. lets not make it too hard on the first try

    def call_PGM(self):


        #call the PGM
        #input('call to pgm')
        #print('in here')
        #print('self.rezStates_minus_by_node.keys()')
        #print(self.rezStates_minus_by_node.keys())
        #input('---')
        while(True): #
            self.debug_check_elem_res_nodes()

            [self.primal_sol,self.dual_exog,self.cur_lp]=self.call_PGM_RMP_solver_from_scratch()#we can do better a different time. lets not make it too hard on the first try
            self.debug_check_elem_res_nodes()
            if self.jy_options['allow_compression']==True:
                if self.cur_lp<self.incumbant_lp-self.jy_options['tolerance_compress']: #if we improve the objective then we will do a compression operator
                    self.apply_compression_operator() #apply compression
                    self.incumbant_lp=self.cur_lp
                    continue
                else:
                    print('skipping incumbant')
            else:
                self.incumbant_lp=self.cur_lp
            self.debug_check_elem_res_nodes()
            did_find_neg_red_cost=False #indicate if we found a negative reduced cost 
            tot_shortest_path_len=0
            self.did_find_new_action=False
            self.did_find_new_state=False
            for my_graph in self.my_PGM_graph_list: #Iterate over all graphs and find the shortest path
                shortest_path, shortest_path_length, ordered_path_rows=my_graph.construct_specific_pricing_pgm(self.dual_exog,self.rez_states_minus_by_node) #construct and call pricing problem
                tot_shortest_path_len=tot_shortest_path_len+min([0,shortest_path_length])
                print('my_graph.l_id')
                print(my_graph.l_id)
                print('shortest_path')
                print(shortest_path)
                print('shortest_path_length')
                print(shortest_path_length)
                print('self.dual_exog')
                print(self.dual_exog)
                print('my_state')

                my_states_in_path=[]
                
                for my_state_id  in shortest_path:
                    my_state=self.my_PGM_graph_list[my_graph.l_id].state_id_to_state[my_state_id]
                    print([my_state.node,my_state.state_vec.toarray()[0][0]])
                    my_states_in_path.append(my_state)
                    #print(my_state.state_vec)
                #input('---')
                if shortest_path_length<-self.jy_options['epsilon']: #if we have a negative reduced cost column we will apply expansion
                    print('abotu to apply expansion shortest_path')
                    
                    #for my_state_id  in shortest_path:
                    #    my_state=self.my_PGM_graph_list[my_graph.l_id].state_id_to_state[my_state_id]
                    #    my_state.pretty_print_state()
                    #input('calling expansion')

                    self.apply_expansion_operator(my_states_in_path,shortest_path, shortest_path_length, ordered_path_rows,my_graph)# do teh expasnion operator
                    did_find_neg_red_cost=True #did find negative reduced cost is set to true
                    #print('done calling expansion')
            self.debug_check_elem_res_nodes()
            print('did_find_neg_red_cost')
            print(did_find_neg_red_cost)
            print('tot_shortest_path_len')
            print(tot_shortest_path_len)
            print('self.cur_lp')
            print(self.cur_lp)
            print('incumbant_lp')
            print(self.incumbant_lp)
            #input('done an iteration of PGM')
            if did_find_neg_red_cost==False: #if we did not find a negative reduced  then we break and we are done 
                # for g_id in self.index_to_graph:
                #     fig = self.visualize_state_action_graph(graph_id=g_id)
                #     plt.show()
                #     fig.savefig(f'state_action_graph_{g_id}.svg', format='svg')
                #     print(f'save graph: {g_id}')
                break
        #input('done call to pgm')


    def apply_expansion_operator(self, my_states_in_path,shortest_path, shortest_path_length, ordered_path_rows, my_graph: Full_Multi_Graph_Object_given_l):
        """Expands the solution by adding new states and actions from the shortest path."""

        #print(f"Applying expansion operator for graph {my_graph}...")
        #print('in apply expansion')
        #print('self.rezStates_minus_by_node.keys()')
        #print(self.rezStates_minus_by_node.keys())
        #print('-----')
        # Step 1: Extract state IDs from the shortest path
        #path_state_ids = set(shortest_path)  # Get all visited state IDs
        
        # Step 2: Map state IDs to actual state objects in the graph
       # path_states = {
       #     state_id: self.pgm_graph_2_rmp_graph[my_graph].state_id_to_state[state_id]
       #     for state_id in path_state_ids
       #     if state_id in self.pgm_graph_2_rmp_graph[my_graph].state_id_to_state
       # }

        # Step 3: Ensure `rezStates_minus_by_node[my_graph]` exists and is structured as a defaultdict(set)
        #if my_graph.l_id not in self.rezStates_minus_by_node:
        #    self.rezStates_minus_by_node[my_graph.l_id] = defaultdict(set)

        # Step 4: Group states by their associated node
        self.debug_check_elem_res_nodes()

        for my_state in my_states_in_path:
            my_node = my_state.node  # Assuming each state object has a `node` attribute
            
            if my_state not in self.rez_states_minus_by_node[my_graph.l_id][my_node]:
                self.did_find_new_state=True
                debug_on=True
                if debug_on==True:
                    tmp_list=[my_state]
                    do_flag=self.is_state_set_subset(self.rez_states_minus_by_node[my_graph.l_id][my_node],tmp_list)
                    if do_flag==True:
                        input('error here')
                self.rez_states_minus_by_node[my_graph.l_id][my_node].add(my_state)
                self.rez_states_minus_by_graph[my_graph.l_id].add(my_state)
                #print('adding state ')
                #input('---')
                my_state.pretty_print_state()
        # Step 5: Extract used actions from ordered path rows
        #if not hasattr(self, "rez_actions"):  # Ensure `res_actions` exists
        #    self.rez_actions = set()
        for _, _, my_action in ordered_path_rows:
            if my_action:  # Ensure the action is not None
                #my_action=self.action_id_2_actions[my_action_id]
                if type(my_action)!=Action:
                    print('my_action')
                    print(my_action)
                    input('this needs to be an action')
                if my_action not in self.rez_actions_minus:
                    self.did_find_new_action=True
                    self.rez_actions_minus.add(my_action)
        if self.did_find_new_action==False and self.did_find_new_state==False:

            input('error nothing added ')
        self.debug_check_elem_res_nodes()

        # Step 6: Update `res_states_minus` as the union of all `res_states_minus_by_graph`
        #self.res_states_minus_by_graph[my_graph.l_id] = set().union(*self.rezStates_minus_by_node[my_graph.l_id].values())
        #self.rez_states_minus_by_graph=dict()# = set().union(*self.rezStates_minus_by_node[my_graph.l_id].values())

        #for g_id in self.index_to_graph:
        #    self.rez_states_minus_by_graph[g_id]=set([])
        #    for n in self.rez_states_minus_by_node[g_id]:
        #       for s in self.rez_states_minus_by_node[g_id][n]:
        #           self.rez_states_minus_by_graph[g_id].add(s)
        self.debug_check_elem_res_nodes()
        #print('at end of thies')
        #print('self.rezStates_minus_by_node.keys()')
        #print(self.rezStates_minus_by_node.keys())
        #print('-----')
        #input('----')


    def apply_compression_operator(self):
        """Applies the compression operator to filter active variables from the LP solution."""
        
        print('Applying compression operator...')

        # Step 1: Extract all active variables from the LP primal solution
        active_vars = {var_name for var_name, value in self.primal_sol.items() if value > 1e-6}

        # Step 2: Separate actions and edges
        self.rez_actions_minus = set()  # Set of selected actions
        active_edges = set()  # Set of selected edges (g, s1, s2)

        for var_name in active_vars:
            if var_name[0] == "eq_act_var":  # Action variable format: ('eq_act_var', g, eq_class, action)
                _, g, eq_class, action_id = var_name  # Extract components
                my_action=self.action_id_2_actions[action_id]
                if type(my_action)!=Action:
                    print('type(action)')
                    print(type(my_action))
                    input('error here')
                self.rez_actions_minus.add(my_action)  # Store the action
                _, g, eq_class, action_id = var_name  # Extract components
                my_action=self.action_id_2_actions[action_id]
                if type(my_action)!=Action:
                    print('type(action)')
                    print(type(my_action))
                    input('error here')
                self.rez_actions_minus.add(my_action)  # Store the action

            elif var_name[0] == "edge":  # Edge variable format: ('edge', g, s1, s2)
                _, g, s1, s2 = var_name  # Extract graph and states
                active_edges.add((g, s1, s2))
        
        if self.the_null_action in self.rez_actions_minus:
            self.rez_actions_minus=self.rez_actions_minus.remove(g.null_action)
        # Step 3: Ensure `rezStates_minus_by_node[g]` exists as a defaultdict(set)
        #self.rez_states_minus_by_node = defaultdict(lambda: defaultdict(list))  # Nested defaultdict for automatic list initialization
        self.rez_states_minus=set()
        # {g: defaultdict(set) for g in self.pgm_graph_2_rmp_graph}  

        # Step 4: Update `rezStates_minus_by_node[g][node]` to include states from active edges
        for g_id, s1_id, s2_id in active_edges:
            
            
            g=self.index_to_graph[g_id]
            if s1_id not in self.pgm_graph_2_rmp_graph[g].state_id_to_state:
                print('error here')
                for g2_id in self.index_to_graph:
                    g2=self.index_to_graph[g2_id]
                    if s1_id in self.pgm_graph_2_rmp_graph[g2].state_id_to_state:
                        print('found here')
                        print(g2_id)
                        input('---')
                    else:
                        print('not in here')
                        print(g2_id)

                #input('here')
            state1 = self.pgm_graph_2_rmp_graph[g].state_id_to_state[s1_id]  # Retrieve state object
            state2 = self.pgm_graph_2_rmp_graph[g].state_id_to_state[s2_id]  # Retrieve state object

            if state1.node!=state2.node:
                #self.rez_states_minus_by_node[g_id][state1.node].add(state1)
                #self.rez_states_minus_by_node[g_id][state2.node].add(state2)
                self.rez_states_minus.add(state1)
                self.rez_states_minus.add(state2)
        self.make_rez_states_minus_by_node
        #compute_res_states,
        #self.res_states_minus= set().union(*self.res_states_minus_by_graph.values())

    def debug_exper(self,B1,B2):

        #consider 2 sets  B1 and B2
        # i will do the following tests and not the inconsistency
        #first i show the types
        print('type(B1)')
        print(type(B1))
        print('type(B2)') 
        print(type(B2)) 
        #now I show that all elements of B1 are in B2
        for s1 in B1:
            if s1 not in B2:
                input('error here')
        #now i count the number of times that statevec's first element is two for each
        count_b1=0
        for s1 in B1:
            if  s1.state_vec.toarray()[0][0]==2:
                count_b1+=1
        count_b2=0
        for s2 in B2:
            if  s2.state_vec.toarray()[0][0]==2:
                count_b2+=1  
        #now lets compare them
        if count_b2!=count_b1:
            print('count_b1')
            print(count_b1)
            print('count_b2')
            print(count_b2)
            input('this is clearly wrong ')
        print('count_b2')
        print(count_b2)
        print('count_b1')
        print(count_b1)
        input('done test')

    def make_rez_states_minus_from_by_nodes(self):
        self.rez_states_minus=set()
        for g_id in self.rez_states_minus_by_node:
            for n in self.rez_states_minus_by_node[g_id]:
                for s in self.rez_states_minus_by_node[g_id][n]:
                    self.rez_states_minus.add(s)

    def DEBUG_count_by_cap_rem_2(self):
        count_b2=0
        for  s in self.rez_states_minus:
            if  s.state_vec.toarray()[0][0]==2:
                count_b2+=1  
        return count_b2
    def return_rez_states_minus_and_res_actions(self):
    # I am trying to fgigure out hwy throught the code I am having trouble with comparison operators. 
        #  Things like saying that a given state is present in a set of states.  
        # this is really slowing me down.  
        # i want to find out how to fix this type of issue.  
        #lets see an instance of this issue that I created
        #this is an artificial issue.  DOnt focus on making the code correct.  Focus on telling me why this issue is occuring
        #I have set self.states_used_sol
        #I also have a set self.res_states_minus
        # i can print out the two sets 
        # i can see that some elements in self.states_used_sol are not present in self.res_states_minus
        # specifically these are terms iwth state vector with a first term other than two.  
        #however my comparison operators seem to say taht the given elements are present
        #please tell me why
        #I see that self.states_used_sol has elements

        
                    #ids_in_state_res_minus.add(s.state_id)
        self.make_rez_states_minus_from_by_nodes()
        #self.debug_exper(self.states_used_sol,self.res_states_minus)
        #for s in self.states_used_sol:
        #    self.res_states_minus.add(s)
        return self.rez_states_minus,self.rez_actions_minus

    def call_PGM_RMP_solver_from_scratch(self):
        """Constructs and initializes the RMP solver from scratch."""
        # Step 1: Initialize the RMP graphs
        self.pgm_graph_2_rmp_graph:DefaultDict[Full_Multi_Graph_Object_given_l,RMP_graph_given_l] = defaultdict()  # Dictionary to store RMP graphs
        #print('self.rezStates_minus_by_node.keys()')
        #print(self.rezStates_minus_by_node.keys())
        #input('---')
        self.l_id_2_active_graph=dict()
        for l_id in self.index_to_graph:#.items():
            g=self.index_to_graph[l_id]
            my_states_g_by_node = self.rez_states_minus_by_node[l_id]
            self.pgm_graph_2_rmp_graph[g] = RMP_graph_given_l(g, my_states_g_by_node, self.rez_actions_minus, self.dominated_actions,self.the_null_action,self.action_id_2_actions)
            if len(my_states_g_by_node)>0.5:
                self.pgm_graph_2_rmp_graph[g].initialize_system()  # Initialize RMP graph
                self.l_id_2_active_graph[l_id]=True
            else:
                self.l_id_2_active_graph[l_id]=False
        # Step 2: Initialize variables and constraints
        self.all_vars = []  # List to store all variables
        self.all_con_names = set()  # Set of all constraint names
        self.lbCon = defaultdict(float)  # Lower bounds on constraints
        self.ubCon = defaultdict(float)  # Upper bounds on constraints

        # Step 3: Create exogenous constraints
        for exog_num in range(self.prob_RHS.size):
            exog_name = ('exog', exog_num)
            self.all_con_names.add(exog_name)
            self.lbCon[exog_name] = self.prob_RHS[exog_num]  # Set lower bound
            #self.ubCon[exog_name] = np.inf  # Upper bound is infinity

        # Step 4: Create non-exogenous constraints
        for g, rmp_graph in self.pgm_graph_2_rmp_graph.items():
            if self.l_id_2_active_graph[g.l_id]==False:
                continue
            for my_eq in rmp_graph.equiv_class_2_s1_s2_pairs:
                non_exog_name = ('eq_con', my_eq, g.l_id)
                self.all_con_names.add(non_exog_name)

                self.ubCon[non_exog_name] = 0
                self.lbCon[non_exog_name] = 0

        # Step 5: Create flow conservation constraints
        for g, rmp_graph in self.pgm_graph_2_rmp_graph.items():
            if self.l_id_2_active_graph[g.l_id]==False:
                continue
            for my_node in rmp_graph.resStates_minus_by_node:
                for my_state in rmp_graph.resStates_minus_by_node[my_node]:
                    if not my_state.is_source and not my_state.is_sink:
                        non_exog_name = ('flow_con', my_state.state_id, g.l_id)
                        self.all_con_names.add(non_exog_name)

                        self.ubCon[non_exog_name] = 0
                        self.lbCon[non_exog_name] = 0
        # Step 6: Create variables and associated actions
        for g, rmp_graph in self.pgm_graph_2_rmp_graph.items():
            #input('1  i should make it here lots of times')
            if self.l_id_2_active_graph[g.l_id]==False:
                continue
            for my_eq in rmp_graph.equiv_class_2_s1_s2_pairs:
                #input('2 i should make it here lots of times')

                for my_act in rmp_graph.equiv_class_2_actions[my_eq]:
                    #input('3  i should make it here lots of times')
                    my_cost = my_act.cost  # Get cost
                    my_exog = my_act.Exog_vec  # Get exogenous vector
                    my_contrib_dict = defaultdict()  # Dictionary for contributions

                    # Use precomputed nonzero indices for efficiency
                    for exog_num in my_act.non_zero_indices_exog:
                        exog_name = ('exog', exog_num)
                        my_contrib_dict[exog_name] = my_act.Exog_vec[exog_num]

                    # Create constraint for the equivalence class
                    non_exog_name = ('eq_con', my_eq, g.l_id)
                    my_contrib_dict[non_exog_name] = -1
                    
                    # Define variable name and store it
                    my_name = ('eq_act_var', g.l_id, my_eq, my_act.action_id)
                    #TODO: remove my_exog here
                    #new_var = jy_var(my_cost, my_exog, my_contrib_dict, my_name)
                    new_var = jy_var(my_cost, my_contrib_dict, my_name)
                    self.all_vars.append(new_var)

                # Step 7: Create variables for state transitions (edges)
                for (s1, s2) in rmp_graph.equiv_class_2_s1_s2_pairs[my_eq]:
                    my_cost = 0
                    my_exog = None  # No exogenous contribution for edges
                    my_contrib_dict = defaultdict(float)

                    non_exog_name = ('eq_con', my_eq, g.l_id)
                    my_contrib_dict[non_exog_name] = 1
                    
                    # Flow conservation constraints
                    
                    if not s1.is_source:
                        flow_in_name_exog_name = ('flow_con', s1.state_id, g.l_id)
                        my_contrib_dict[flow_in_name_exog_name] = 1
                        
                    if not s2.is_sink:
                        flow_out_name_exog_name = ('flow_con', s2.state_id, g.l_id)
                        my_contrib_dict[flow_out_name_exog_name] = -1
                        

                    # Define variable name and store it
                    my_name = ('edge', g.l_id, s1.state_id, s2.state_id)
                    if g.l_id>0.5 and s1.node>0 and s2.node>0 :
                        if s2.state_vec.toarray()[0][0]==s1.state_vec.toarray()[0][0]:
                            print('making edge')
                            print('s1.node,s2.node')
                            print([s1.node,s2.node])
                            print('s1.state_vec.toarray()')
                            print(s1.state_vec.toarray())
                            print('s2.state_vec.toarray()')
                            print(s2.state_vec.toarray())
                            input('yo error here')
                    #TODO: remove my_exog here
                    new_var = jy_var(my_cost, my_contrib_dict, my_name)
                    if s1.state_id not in self.pgm_graph_2_rmp_graph[g].state_id_to_state:
                        print('s1.state_id')
                        print(s1.state_id)
                        print('s1.l_id')
                        print(s1.l_id)
                        print('s1.node')
                        print(s1.node)
                        print('self.pgm_graph_2_rmp_graph[g].l_id')
                        print(self.pgm_graph_2_rmp_graph[g].l_id)
                        input('errror here1 ')
                    if s2.state_id not in self.pgm_graph_2_rmp_graph[g].state_id_to_state:
                        print('s2.state_id')
                        print(s2.state_id)
                        print('s2.node')
                        print(s2.node)
                        print('s2.l_id')
                        print(s2.l_id)
                        print('self.pgm_graph_2_rmp_graph[g].l_id')
                        print(self.pgm_graph_2_rmp_graph[g].l_id)
                        input('errror here2 ')
                    self.all_vars.append(new_var)
                    #print('new_var')
                    #print(new_var)
                    #input('----')
        


        primal_solution, dual_solution, optimal_value = self.solve_with_pulp(self.all_vars,self.all_con_names,self.lbCon,self.ubCon)
        self.debug_check_primal_solution_match(primal_solution)
        self.debug_check_primal_exog_feas(primal_solution)
        self.decode_sol_2_paths(primal_solution)

        dual_exog=np.zeros(self.prob_RHS.size)
        for exog_num in range(self.prob_RHS.size):
            exog_name = ('exog', exog_num)
            exog_name_aug="LowerBound_"+str(exog_name)
            #print('exog_name')
            #print(exog_name)
            #print('exog_name_aug')
            #print(exog_name_aug)
            #print('exog_name_aug in dual_solution')
            #print(exog_name_aug in dual_solution)
            dual_exog[exog_num]=dual_solution[exog_name_aug]
        
        return primal_solution, dual_exog, optimal_value

    def solve_with_pulp(self, jy_vars, all_con_names, lbCon, ubCon):
        # Step 1: Create a PuLP minimization problem
        prob = pl.LpProblem(name="OptimizationProblem", sense=pl.LpMinimize)
        
        # Step 2: Create PuLP variables
        pulp_vars = {}
        for var in jy_vars:
            var_name = var.my_name
            pulp_vars[var_name] = pl.LpVariable(name=str(var_name), lowBound=0, cat='Continuous')
        
        # Step 3: Define the Objective Function (Minimize Cost)
        objective = pl.lpSum(var.my_cost * pulp_vars[var.my_name] for var in jy_vars)
        prob += objective
        
        # Step 4: Add Constraints
        constraint_dict = {}  # Store constraint objects for dual values
        constraint_mapping = {}  # Map from original constraint name to the actual PuLP constraint name
        
        for con_name in all_con_names:
            # Compute constraint sum from contributions
            constraint_expr = pl.lpSum(var.my_contrib_dict.get(con_name, 0) * pulp_vars[var.my_name] for var in jy_vars)
            
            # Apply lower and upper bounds if they exist
            if con_name in lbCon:
                constraint = constraint_expr >= lbCon[con_name]
                # Create a much simpler constraint name using a counter
                constraint_name = f"LB_{len(constraint_dict)}"
                prob += (constraint, constraint_name)
                constraint_dict[constraint_name] = constraint
                constraint_mapping[f"LowerBound_{con_name}"] = constraint_name
                
            if con_name in ubCon:
                constraint = constraint_expr <= ubCon[con_name]
                # Create a much simpler constraint name using a counter
                constraint_name = f"UB_{len(constraint_dict)}"
                prob += (constraint, constraint_name)
                constraint_dict[constraint_name] = constraint
                constraint_mapping[f"UpperBound_{con_name}"] = constraint_name
        
        # Step 4.5: Print the model formulation
        #self.print_pulp_formulation(prob)
        
        # Step 5: Solve the problem
        print('starting lp call')
        prob.solve()
        print('done lp call')

        # Step 6: Extract solutions
        primal_solution = {}
        for var_name, pulp_var in pulp_vars.items():
            primal_solution[var_name] = pulp_var.value()
        
        # Extract dual values
        dual_solution = {}
        
        if prob.status == 1:  # If the problem was solved optimally
            # Print all constraint names that PuLP knows about
            #print("Available constraints in PuLP:", list(prob.constraints.keys()))
            
            # First, get all the duals using the simplified names we created
            temp_duals = {}
            for simplified_name in constraint_dict.keys():
                try:
                    if simplified_name in prob.constraints:
                        temp_duals[simplified_name] = prob.constraints[simplified_name].pi
                    else:
                        # Try a direct lookup in constraints dictionary
                        found = False
                        for name in prob.constraints:
                            if simplified_name in name:  # Check if our simplified name is part of the actual name
                                temp_duals[simplified_name] = prob.constraints[name].pi
                                found = True
                                print(f"Found constraint {simplified_name} as {name}")
                                break
                        
                        if not found:
                            print(f"Warning: Could not find constraint: {simplified_name}")
                            temp_duals[simplified_name] = 0
                except Exception as e:
                    print(f"Error getting dual for {simplified_name}: {e}")
                    temp_duals[simplified_name] = 0
            
            # Now map back to the original constraint names format
            for original_name, simplified_name in constraint_mapping.items():
                dual_solution[original_name] = temp_duals.get(simplified_name, 0)
        else:
            # If the problem wasn't solved optimally, set all duals to 0
            for original_name in constraint_mapping.keys():
                dual_solution[original_name] = 0
        
        dual_sol = np.array(list(dual_solution.values()))
        
        # Get optimal objective value
        optimal_value = pl.value(prob.objective)

        if optimal_value>self.lp_before_operations+.0001 :
            input('error here objective went up')
        else:
            self.lp_before_operations=optimal_value
        #self.print_pulp_formulation(prob)
        #input('---')
        #
        #print('starting outputting solution ')
        #for var_name in primal_solution:
        #    # Get the solution value for each variable
        #    if np.abs(primal_solution[var_name])>.0001:
        #        print(var_name)
        #        print(primal_solution[var_name])
        #        self.help_print_var(var_name)
#
 #       input('outputting solution')
 

        return primal_solution, dual_solution, optimal_value

    def construct_and_solve_lp(self, jy_vars, all_con_names, lbCon, ubCon):
        """
        Constructs and solves an Xpress Linear Program (LP), returning primal and dual solutions.
        """
        import xpress as xp
        # Step 1: Initialize Xpress problem
        xp.init('C:/xpressmp/bin/xpauth.xpr')  # Ensure Xpress is initialized correctly
        prob = xp.problem()

        # Step 2: Create Xpress variables
        xpress_vars = {}

        for var in jy_vars:
            var_name = var.my_name  # Ensure variable name is a string
            xpress_vars[var_name] = xp.var(vartype=xp.continuous, name=str(var_name),lb=0,ub=float("inf"))  # Create variable
        prob.addVariable(list(xpress_vars.values()))  # Add variables to the problem

        # Step 3: Define the Objective Function (Minimize Cost)
        objective = xp.Sum(var.my_cost * xpress_vars[var.my_name] for var in jy_vars)
        prob.setObjective(objective, sense=xp.minimize)

        # Step 4: Add Constraints
        constraint_dict = {}  # Store constraint objects for dual values
        for con_name in all_con_names:
            # Compute constraint sum from contributions
            constraint_expr = xp.Sum(var.my_contrib_dict.get(con_name, 0) * xpress_vars[var.my_name] for var in jy_vars)

            # Apply lower and upper bounds if they exist
            if con_name in lbCon:
                constraint = xp.constraint(constraint_expr >= lbCon[con_name], name=f"LowerBound_{con_name}")
                prob.addConstraint(constraint)
                constraint_dict[f"LowerBound_{con_name}"] = constraint

            if con_name in ubCon:
                constraint = xp.constraint(constraint_expr <= ubCon[con_name], name=f"UpperBound_{con_name}")
                prob.addConstraint(constraint)
                constraint_dict[f"UpperBound_{con_name}"] = constraint

        


        prob.solve()
        
        primal_solution = {}

        # Iterate through the xpress_vars dictionary
        for var_name, xp_var in xpress_vars.items():
            # Get the solution value for each variable
            primal_solution[var_name] = prob.getSolution(xp_var)


        dual_solution = {}

        # Iterate through the constraint_dict dictionary
        for con_name, constraint in constraint_dict.items():
            # Get the dual value for each constraint
            dual_solution[con_name] = prob.getDual(constraint)

        dual_sol = np.array(list(dual_solution.values()))

        optimal_value = prob.getObjVal()



        return primal_solution, dual_sol, optimal_value

 
    def print_pulp_formulation(self,prob):

        """

        Print the formulation of a PuLP model with variable names, coefficients, and RHS values.

        Args:

            prob: A PuLP problem object

        """

        print("=" * 80)

        print("MODEL FORMULATION")

        print("=" * 80)

        # Print objective function

        print("\nOBJECTIVE FUNCTION:")

        if prob.sense == 1:  # Minimize

            print("Minimize:")

        else:

            print("Maximize:")

        obj_terms = []

        for var, coeff in prob.objective.items():

            if coeff != 0:  # Only include non-zero coefficients

                if abs(coeff) == 1:

                    sign = "-" if coeff < 0 else "+"

                    obj_terms.append(f"{sign} {var.name}")

                else:

                    sign = "-" if coeff < 0 else "+"

                    obj_terms.append(f"{sign} {abs(coeff)} {var.name}")

        # Format the objective function nicely

        obj_str = " ".join(obj_terms)

        # Replace leading "+" with empty string if it exists

        obj_str = obj_str[2:] if obj_str.startswith("+ ") else obj_str

        print(f"    {obj_str}")

        # Print constraints

        print("\nCONSTRAINTS:")

        for name, constraint in prob.constraints.items():

            con_terms = []

            # Get the left-hand side terms

            for var, coeff in constraint.items():

                if coeff != 0:  # Only include non-zero coefficients

                    if abs(coeff) == 1:

                        sign = "-" if coeff < 0 else "+"

                        con_terms.append(f"{sign} {var.name}")

                    else:

                        sign = "-" if coeff < 0 else "+"

                        con_terms.append(f"{sign} {abs(coeff)} {var.name}")

            # Format the constraint left-hand side

            con_str = " ".join(con_terms)

            # Replace leading "+" with empty string if it exists

            con_str = con_str[2:] if con_str.startswith("+ ") else con_str

            # Determine constraint sense and right-hand side

            sense = ""

            rhs = 0

            if constraint.sense == 1:  # ≤

                sense = "<="

                rhs = constraint.constant * -1

            elif constraint.sense == -1:  # ≥

                sense = ">="

                rhs = constraint.constant * -1

            else:  # ==

                sense = "="

                rhs = constraint.constant * -1

            print(f"[{name}]: {con_str} {sense} {rhs}")

        # Print variable bounds

        print("\nVARIABLE BOUNDS:")

        for var in prob.variables():

            lb = var.lowBound if var.lowBound is not None else "-inf"

            ub = var.upBound if var.upBound is not None else "+inf"

            if lb == ub:

                print(f"{var.name} = {lb}")

            else:

                print(f"{lb} <= {var.name} <= {ub}")

        # Print variable types

        print("\nVARIABLE TYPES:")

        for var in prob.variables():

            var_type = "Continuous"

            if var.cat == pl.LpInteger:

                var_type = "Integer"

            elif var.cat == pl.LpBinary:

                var_type = "Binary"

            print(f"{var.name}: {var_type}")

        print("=" * 80)

    def visualize_state_action_graph(self, figsize=(20, 10), title="State-Action Graph", graph_id=None):
        """
        Visualize a state-action graph with states arranged horizontally by node values (left to right).
        Multiple states with the same node value are arranged vertically.
        States are labeled with the value of state.state_vec.toarray()[0,0].
        
        Args:
            figsize (tuple): Figure size (width, height)
            title (str): Title for the visualization
            graph_id (int, optional): If provided, only visualize states from this graph
            
        Returns:
            matplotlib figure
        """
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Filter states by graph_id if specified
        if graph_id is not None:
            states_to_use = [s for s in self.rez_states_minus if s.l_id == graph_id]
        else:
            states_to_use = list(self.rez_states_minus)
        
        # Dictionary to store the original state objects by state_id for later reference
        state_objects = {}
        
        # Add states as nodes with their attributes
        for state in states_to_use:
            # Extract the state vector value for labeling
            state_vec_value = state.state_vec.toarray()[0, 0]
            
            G.add_node(state.state_id, 
                    node=state.node,
                    l_id=state.l_id,
                    state_vec_value=state_vec_value)
            
            # Store original state object for edge creation
            state_objects[state.state_id] = state
        
        # Add action connections from flow constraints
        # Collect all potential state pairs that could form an edge
        edges_to_add = []
        
        # For each graph in the model
        for g_id in self.index_to_graph:
            # Skip if we're only visualizing a specific graph and this isn't it
            if graph_id is not None and g_id != graph_id:
                continue
                
            g = self.index_to_graph[g_id]
            
            # Skip if this graph has no RMP representation
            if not hasattr(self, 'pgm_graph_2_rmp_graph') or g not in self.pgm_graph_2_rmp_graph:
                continue
            
            rmp_graph = self.pgm_graph_2_rmp_graph[g]
            
            # For each equivalence class in the RMP graph
            for eq_class in rmp_graph.equiv_class_2_s1_s2_pairs:
                # For each state pair in this equivalence class
                for s1, s2 in rmp_graph.equiv_class_2_s1_s2_pairs[eq_class]:
                    # Add an edge from s1 to s2 if their state_ids are in our graph
                    if s1.state_id in G.nodes and s2.state_id in G.nodes:
                        edges_to_add.append((s1.state_id, s2.state_id, eq_class))
        
        # Add all collected edges to the graph
        for s1_id, s2_id, eq_class in edges_to_add:
            G.add_edge(s1_id, s2_id, eq_class=eq_class)
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Position nodes in a horizontal layout based on their node values
        # Group states by node value
        nodes_by_level = defaultdict(list)
        for node_id, node_data in G.nodes(data=True):
            node_value = node_data['node']
            nodes_by_level[node_value].append(node_id)
        
        # Position nodes horizontally by their node value, with multiple states per level stacked vertically
        pos = {}
        
        # Custom sorting for node values to place source (-1) at far left and sink (-2) at far right
        sorted_levels = []
        
        # First add source node (-1) if it exists
        if -1 in nodes_by_level:
            sorted_levels.append(-1)
        
        # Then add all other nodes except sink in ascending order
        for level in sorted([l for l in nodes_by_level.keys() if l not in [-1, -2]]):
            sorted_levels.append(level)
        
        # Finally add sink node (-2) if it exists
        if -2 in nodes_by_level:
            sorted_levels.append(-2)
        
        for i, level in enumerate(sorted_levels):
            nodes = nodes_by_level[level]
            # Sort nodes at this level by their state_id for consistent positioning
            nodes.sort()
            
            # Position each node at this level
            for j, node_id in enumerate(nodes):
                # Horizontal position determined by node value
                x = i
                # Vertical position spaces nodes at the same level
                # Center nodes vertically
                y = j - (len(nodes) - 1) / 2
                pos[node_id] = (x, y)
        
        # Draw nodes (states)
        nx.draw_networkx_nodes(G, pos, 
                            node_size=2000, 
                            node_color='lightblue',
                            alpha=0.8,
                            edgecolors='black',
                            ax=ax)
        
        # Draw edges (actions)
        nx.draw_networkx_edges(G, pos,
                            width=1.5,
                            alpha=0.7,
                            edge_color='gray',
                            arrowsize=20,
                            connectionstyle='arc3,rad=0.1',  # Curved edges for better visibility
                            ax=ax)
        
        # Add state labels with only node number and state_vec value (no state ID)
        labels = {node: f"node:{data['node']}\ncap:{int(data['state_vec_value'])}" 
                for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)
        
        # Remove axis
        ax.set_axis_off()
        
        # Add title
        graph_title = title
        if graph_id is not None:
            graph_title += f" (Graph {graph_id})"
        plt.title(graph_title, fontsize=15)
        
        # Add legend
        state_patch = mpatches.Patch(color='lightblue', label='States')
        action_patch = mpatches.Patch(color='gray', label='Actions')
        plt.legend(handles=[state_patch, action_patch], loc='upper right')
        
        plt.tight_layout()
        return fig

    
