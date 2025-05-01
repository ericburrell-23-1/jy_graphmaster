import bisect
import numpy as np
from numpy import zeros, ones
from scipy.sparse import csr_matrix
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from src.common.state import State
from src.common.action import Action
from src.common.pgm_approach import Route
from src.common.jy_label import jy_label
from src.common.jy_eff_fronteir import jy_efficient_frontier
from src.common.jy_sortedObject_list import jy_sortedObject_list
from itertools import permutations
from tqdm import tqdm
import time
class jy_fast_pricing():
    """
    Class implementing a label-setting algorithm for finding minimum reduced cost paths
    with a given dual value.
    """

    def __init__(self, 
                 all_actions, 
                 action_dict, 
                 can_group,
                 edges,
                 preferred_actions,
                 distance,
                 dual_vec, 
                 init_res_state, 
                 max_actions_in_route, 
                 actions_of_node, 
                 all_nodes,
                 neighbors,
                 benefit_group,
                 benefit_group_cost,
                 jy_opt):
        """
        Initialize the jy_fast_pricing algorithm.
        
        Args:
            all_actions: All actions with (origin node, destination node) as key, Action class as object
            dual_vec: Dual vector
            init_res_state: Initial states to expand
            max_actions_in_route: Maximum actions allowed in a route
            actions_of_node: Dict with node as key, list of actions originating from this node as value
            all_nodes: All nodes
            jy_opt: Options dictionary
        """
        self.all_actions = all_actions
        self.action_dict = action_dict
        self.can_group = can_group
        self.edges = edges
        self.preferred_actions = preferred_actions
        self.distance = distance
        self.forbidden_nodes=set()
        self.dual_vec = dual_vec.copy()
        self.dual_vec_orig = dual_vec.copy()
        self.init_res_state = init_res_state
        self.max_actions_in_route = max_actions_in_route
        self.actions_of_node = actions_of_node
        self.all_nodes = all_nodes
        self.neighbors = neighbors
        self.benefit_group = benefit_group
        self.benefit_group_cost = benefit_group_cost
        self.jy_opt = jy_opt
        self.jy_opt['use_load_ai_fast']=True
        self.num_cus = int((len(self.all_nodes)-2)/3)
        self.pickup_node = self.all_nodes[1:self.num_cus+1]
        self.dropoff_node = self.all_nodes[self.num_cus+1:2*self.num_cus+1]
        self.skip_node = self.all_nodes[2*self.num_cus+1:-1]
        # Initialize additional attributes needed for the algorithm
        self.expandable_labels = jy_sortedObject_list()
        self.efficient_frontier = jy_efficient_frontier(all_nodes)
        
        # Prepare action reduced costs and find lowest reduced cost
        self.action_2_red_cost_dict = {}
        self.option_do_min_term = False
        self.lowest_action_contrib_red_cost = float('inf')
        for u, benefit_group in self.benefit_group.items():
            updated_benefits = {}
            if len(benefit_group) > self.jy_opt['k_benefit_group']:
                for v, value in benefit_group.items():
                    updated_benefits[v] = self.benefit_group_cost[u][v] - self.dual_vec[u-1] - self.dual_vec[v-1]
                self.benefit_group[u] = dict(sorted(updated_benefits.items(), key=lambda item: item[1]))
        self.pre_process__partition_actions()
        self.initiate_RCP_d()
        self.initiate_RCP_u()
        self.initiate_RCP_u_2()
        print('initialization')

    def label_2_tuple(self,my_lab):
        #order of expansion 
        f1=my_lab.num_dropoffs_needed
        f2=my_lab.red_cost
        f3=-len(my_lab.all_nodes_ordered)
        f4=my_lab.all_nodes_ordered[-1]
        f5=my_lab.lb
        f6=np.random.randint(10000000)
        
        my_tup=tuple([f1,f2,f3,f4,f5,f6])

        return my_tup
    def run(self):
        """
        Execute the pricing algorithm to find routes with negative reduced cost.
        
        Returns:
            List of Route objects with negative reduced cost
        """
        # Reset state for a new run
        self.expandable_labels = jy_sortedObject_list()
        self.efficient_frontier = jy_efficient_frontier(self.all_nodes)
        
        # Find minimum reduced cost paths
        for threshold,this_preferred_actions in self.preferred_actions.items():

            routes = self.find_min_reduced_cost_path(this_preferred_actions)
            if len(routes)>0:
                break
        
        # Return the routes found
        #print(self.expanede_label[:10])
        #breakpoint()
        return routes

    def _compute_action_reduced_costs(self):
        """
        Compute reduced costs for all actions and identify the lowest reduced cost.
        """
        #forbidden_nodes=[]
        # print('dual_vec')
        # print(self.dual_vec)

        #print('type(self.dual_vec)')
        #print(type(self.dual_vec))
        if isinstance(self.dual_vec,list):
            self.dual_vec = np.array(self.dual_vec,dtype=float)
        ignore_vals=np.nonzero(np.array(self.dual_vec)<0.0001)[0]


        if len(ignore_vals)>0:
            pickup_forget=ignore_vals+1
            dropoff_forget=len(self.pickup_node)+pickup_forget
            self.forbidden_nodes.update(pickup_forget)
            self.forbidden_nodes.update(dropoff_forget)
  
    def initiate_RCP_d(self):
        max_width = self.jy_opt['max_pickups_in_a_route']
        self.rcp_d_partial = defaultdict()
        for cur_loc in self.all_nodes:
            this_rcp = defaultdict()
            for d in self.pickup_node:
                drop_off = d + len(self.pickup_node)
                if (cur_loc,drop_off) in self.action_dict.keys():
                    self.rcp_d_partial[(cur_loc,drop_off)] = self.action_dict[(cur_loc,drop_off)][0].cost/max_width

    def initiate_RCP_u(self):
        max_width = self.jy_opt['max_pickups_in_a_route']
        self.rcp_u_partial = defaultdict()
        for u in self.pickup_node:
            self.rcp_u_partial[u] = self.action_dict[(u,u+len(self.pickup_node))][0].cost/max_width
        self.rcp_u_partial = dict(sorted(self.rcp_u_partial.items(), key=lambda item: item[1], reverse=True))

    def initiate_RCP_u_2(self):
        self.rcp_u_partial_2 = defaultdict()
        for k in range(1,self.jy_opt['max_pickups_in_a_route']+1):
            for u in self.pickup_node:
                self.rcp_u_partial_2[(k,u)] = self.action_dict[(u,u+len(self.pickup_node))][0].cost/k
            
    def initialize_source_label(self):
        """
        Initialize the label at the source node.
        """
        # Create source label with empty actions list, initial state, zero reduced cost and cost
        source_label = jy_label(
            my_actions_ordered=[],
            my_states_ordered=[self.init_res_state],
            red_cost=0,
            cost=0,
            parent_label=None,
            dual_vec=self.dual_vec,
            max_actions_in_route=self.max_actions_in_route,
            lowest_action_contrib_red_cost=self.lowest_action_contrib_red_cost,
            actions_of_node=self.actions_of_node,
            action_dict = self.action_dict,
            jy_opt=self.jy_opt,
            cus_num=self.num_cus,
            pickup_nodes=self.pickup_node,
            dropoff_nodes=self.dropoff_node,
            rcp_u_partial = self.rcp_u_partial,
            rcp_d_partial = self.rcp_d_partial,
            rcp_u_partial_2 = self.rcp_u_partial_2,
            edges = self.edges,
            preferred_actions = self.preferred_actions,
            distance = self.distance
        )

        new_tuple=self.label_2_tuple(source_label)
        # Add to expandable labels with its reduced cost as the key
        #self.expandable_labels.insert(source_label, source_label.red_cost)
        self.expandable_labels.insert(source_label, new_tuple)
        
        # Add to efficient frontier
        self.efficient_frontier.alter_fronteir_given_new_element(source_label)
        
        return source_label

    def update_red_cost_and_lb(self):
        #print('self.dual_vec')
        #print(self.dual_vec)
        #print('np.sum(self.dual_vec)')
        #print(np.sum(self.dual_vec))
        #input('self.dual_vec')
        self._compute_action_reduced_costs()
        debug_on=True
        
        for my_label in self.expandable_labels.objects:
            if my_label.red_cost != np.inf:
                my_label.calculate_red_cost_given_dual(self.dual_vec)
                #my_label.calculate_lb_given_lowest_action_contrib_red_cost(self.lowest_action_contrib_red_cost)
                if self.jy_opt['lb_option'] == 2:
                    my_label.calculate_better_lb_2(self.dual_vec,self.sorted_node_with_k)
                elif self.jy_opt['lb_option'] == 1:
                    my_label.calculate_better_lb(self.dual_vec)
                elif self.jy_opt['lb_option'] == 0:
                    my_label.calculate_lb_given_lowest_action_contrib_red_cost(self.lowest_action_contrib_red_cost)
                elif self.jy_opt['lb_option'] == 'check':
                    lb1 = my_label.calculate_better_lb(self.dual_vec)
                    lb2 = my_label.calculate_better_lb_2(self.dual_vec,self.sorted_node_with_k)
                    if lb1<lb2:
                        input('lb error here')
                else:
                    input('no lb option used')
                tmp=set(my_label.all_nodes_ordered).intersection(self.forbidden_nodes)
                if len(tmp)>0.5:
                    my_label.red_cost=np.inf
                    my_label.lb=np.inf

    def get_lowest_lb(self):
        lowest_lb=np.inf
        for input_label in self.expandable_labels.objects:
            if lowest_lb>input_label.lb:
                lowest_lb=input_label.lb
            #lowest_lb=np.min(lowest_lb,)
            #self.node_2_eff_fronteir[my_node]
        return lowest_lb
    def _update_sorted_node_with_k(self):
        self.sorted_node_with_k = {}
        for num_pickups in range(1,self.jy_opt['max_pickups_in_a_route']+1):
            # Calculate tot_gain for all nodes at once
            tot_gain = {
                u: -self.dual_vec[u-1] + self.rcp_u_partial_2[(num_pickups, u)]
                for u in self.pickup_node
            }
            # Sort once and store the sorted items
            self.sorted_node_with_k[num_pickups] = dict(sorted(tot_gain.items(), key=lambda item: item[1]))
    def find_min_reduced_cost_path(self,prefered_actions):
        """
        Main method to find the minimum reduced cost path following the algorithm in the PDF.
        
        Returns:
            List of routes with negative reduced cost
        """
        # Initialize with source label
        self._compute_action_reduced_costs()
        source_label = self.initialize_source_label()
        all_routes = []
        min_reduce_cost = np.inf
        self.expandable_labels = jy_sortedObject_list()
        #my_tup=tuple([-len(source_label.my_states_ordered),source_label.red_cost,my_noise])
        new_tuple=self.label_2_tuple(source_label)
        self.expandable_labels.insert(source_label, new_tuple)
        # Outer loop as in algorithm
        cur_red_cost = 0
        num_expansion_out=0
        num_expansion_in=0
        alpha = self.jy_opt.get('alpha', 1.0)  # Default to 1.0 or maybe 0.5
        #print('starting the CG process')
        #input('----')
        debug_on=False
        lowest_so_far=np.inf
        route_gen_count=0
        use_completion_on=False
        max_ter = 1
        itr_num = 0
        if self.jy_opt['use_comp_col'] == True:
            max_ter =100000
        self.expanede_label = []
        while itr_num < max_ter:
            itr_num+=1
            # Re-compute bounds based on dual values
            # Remove expandable labels with LB > 0
            num_expansion_out=num_expansion_out+1
            #print('num_expansion_out,num_expansion_in')
            #print(num_expansion_out,num_expansion_in)
            #input('redoing labels')
            self._update_sorted_node_with_k()
            self.update_red_cost_and_lb()
            
            self._remove_labels_with_positive_lb()

            # Update efficient frontier with current set of expandable labels
            for label in self.expandable_labels.objects:
                if label.lb<self.jy_opt['min_dual_val_expand']:
                    self.efficient_frontier.alter_fronteir_given_new_element(label)
            # Check if we can terminate
        
            min_lb = float('inf')
            for label in self.expandable_labels.objects:
                if label.lb < min_lb:
                    min_lb = label.lb
            # print('min_lb')
            # print(min_lb)
            if min_lb >= 0 or len(self.expandable_labels) == 0:
                break
            
            # Inner loop to process expandable labels
            #input('starting inner')
            num_expanded_this_round=0
            incumbant_lb=-np.inf
            print('len(self.expandable_labels)')
            print(len(self.expandable_labels))
            while len(self.expandable_labels) > 0:
                num_expansion_in=num_expansion_in+1
                num_expanded_this_round=num_expanded_this_round+1
                #print('num_expansion_out,num_expansion_in')
                #print([num_expansion_out,num_expansion_in])
                # Pop label with minimum current reduced cost
                if debug_on==True:  
                    my_lb=self.get_lowest_lb()
                    #print('my_lb')
                    #print(my_lb)
                    #input('my_lb')
                    if my_lb<incumbant_lb:
                        print('incumbant_lb')
                        print(incumbant_lb)
                        print('my_lb')
                        print(my_lb)
                        input('error here: my_lb<incumbant_lb')
                    else:
                        incumbant_lb=my_lb
                curr_label = self.expandable_labels.pop()
                self.expanede_label.append(curr_label.all_nodes_ordered)
                if debug_on==True:
                    self.jy_get_compelition(curr_label)
                verbose=True
                if curr_label.lb>self.jy_opt['min_dual_val_expand']:
                    print('curr_label.lb')
                    print(curr_label.lb)
                    print('num_expansion_in')
                    print(num_expansion_in)
                    print('itr_num')
                    print(itr_num)
                    print('curr_label.all_nodes')
                    print(curr_label.all_nodes_ordered)
                    input('error here:curr_label.lb>self.jy_opt')
                if curr_label.lb>self.jy_opt['min_dual_val_expand']:
                    continue
                if 0>0:
                    if verbose==True and num_expansion_in % 100==0:
                        print('incumbant_lb')
                        print(incumbant_lb)
                        print('curr_label.red_cost')
                        print(curr_label.red_cost)
                        print('curr_label.LB')
                        print(curr_label.lb)
                        print('len(curr_label.my_states_ordered)')
                        print(len(curr_label.my_states_ordered))
                        print('curr_label.all_nodes_ordered')
                        print(curr_label.all_nodes_ordered)
                        print('self.forbidden_nodes')
                        print(self.forbidden_nodes)
                        print('len(self.expandable_labels)')
                        print(len(self.expandable_labels))
                        print('num_expansion_in')
                        print(num_expansion_in)
                        print('num_expansion_out')
                        print(num_expansion_out)
                        #input('error: verbose==True and num_expansion_in % 100==0')
                if debug_on==True:
                    #check the lower bound
                    old_lb=curr_label.lb
                    if self.jy_opt['lb_option'] == 2:
                        curr_label.calculate_better_lb_2(self.dual_vec,self.sorted_node_with_k)
                    elif self.jy_opt['lb_option'] == 1:
                        curr_label.calculate_better_lb(self.dual_vec)
                    elif self.jy_opt['lb_option'] == 0:
                        curr_label.calculate_lb_given_lowest_action_contrib_red_cost(self.lowest_action_contrib_red_cost)
                    elif self.jy_opt['lb_option'] == 'check':
                        lb1 = curr_label.calculate_better_lb(self.dual_vec)
                        lb2 = curr_label.calculate_better_lb_2(self.dual_vec,self.sorted_node_with_k)
                        if lb1<lb2:
                            input('lb error here')
                    else:
                        input('no lb option used')
                    if abs(curr_label.lb-old_lb)>.001:
                        input('error here')
                # Generate all possible expansions for this label
                #expanded_labels = curr_label.expand_label_fully()
                if self.jy_opt['poss_action'] == 1:
                    poss_actions  = self.get_actions_from_label(curr_label)
                elif self.jy_opt['poss_action'] == 2:
                    poss_actions = self.get_actions_from_label_2(curr_label,prefered_actions)
                else:
                    input('error')


                # print('old label num')
                # print(len(poss_actions))
                # print('new label num')
                # print(len(poss_actions_2))
                # print('check ')
                #print('len(poss_actions)')
                #print(len(poss_actions))
                did_gen_neg_red_cost=False
                did_gen_possible_expansion=False
                # Process each expanded label
                #print('------')
                #print('------')
                #print('------')
                #print('------')
                #print('------')
                #if curr_label.all_nodes_ordered ==[-1,4,9]:
                #    print('check here')

                #print('check point here')
                for my_act in poss_actions:

                    if my_act.node_head in self.skip_node:
                        continue
                    if curr_label.all_nodes_ordered == [-1, 1, 4]:
                        print('check here')
                    new_label= curr_label.expand_given_action(my_act,self.dual_vec,self.forbidden_nodes)

                    if new_label == None:
                        #print('doing none')
                        continue
                    if self.jy_opt['lb_option'] == 2:
                        new_label.calculate_better_lb_2(self.dual_vec,self.sorted_node_with_k)
                    elif self.jy_opt['lb_option'] == 1:
                        new_label.calculate_better_lb(self.dual_vec)
                    elif self.jy_opt['lb_option'] == 0:
                        new_label.calculate_lb_given_lowest_action_contrib_red_cost(self.lowest_action_contrib_red_cost)
                    elif self.jy_opt['lb_option'] == 'check':
                        lb1 = new_label.calculate_better_lb(self.dual_vec)
                        lb2 = new_label.calculate_better_lb_2(self.dual_vec,self.sorted_node_with_k)
                        if lb1<lb2:
                            input('lb error here')
                    else:
                        input('no lb option used')

                    if use_completion_on:
                        can_complete=self.jy_get_compelition(new_label)


                        if can_complete==False:
                            #print('no completion')
                            continue
                    #print('t3')
                    if my_act.node_head in self.dropoff_node or my_act.node_head==-2:
                        #print('OKY GOOD ')
                        did_gen_possible_expansion=True
                    #print('t4')
                    if new_label.lb>5:
                        continue

                    if new_label.lb<self.jy_opt['min_dual_val_expand']:
                        self.efficient_frontier.alter_fronteir_given_new_element(new_label)


                    if new_label.is_complete_route:
                        lowest_so_far=np.min([lowest_so_far,new_label.red_cost])
                    if new_label.is_complete_route  and new_label.red_cost<-.001: #< new_label.lb/10:
                        #input('making route')
                        route = new_label.convert_2_route()
                        all_routes.append(route)
                        # print('route made')
                        # print('new_label.all_nodes_ordered')
                        # print(new_label.all_nodes_ordered)
                        # print('new_label.all_nodes_ordered')
                        # print('new_label.red_cost')
                        # print(new_label.red_cost)
                        #input('paused')
                        self.dual_vec = self.dual_vec - route.Exog_vec*self.dual_vec_orig*alpha
                        route_gen_count=route_gen_count+1
                        did_gen_neg_red_cost=True
                        break
                        #if alpha>.99:
                        #    continue

                    elif not new_label.is_complete_route:
                        #new_label.calculate_better_lb(self.dual_vec)
                        new_tuple=self.label_2_tuple(new_label)
                        if new_label.lb<self.jy_opt['min_dual_val_expand']:
                            self.expandable_labels.insert(new_label,new_tuple)
                # print('======check for each action expand while loop=======')
                # print(f'time_action_add: {time_action_add}')
                # print(f'time_expand_action:{time_expand_action}, percent:{(time_expand_action/time_action_add)*100} %')
                # print(f'time_create_label_time:{time_create_label_time}, percent:{(time_create_label_time/time_action_add)*100} %')
                # print(f'get_head_state_time:{time_get_head_state}, percent:{(time_get_head_state/time_action_add)*100} %')
                # print(f'time_clip_time:{time_clip_time}, percent:{(time_clip_time/time_action_add)*100} %')
                # print(f'time_cal_lb:{time_cal_lb}, percent:{(time_cal_lb/time_action_add)*100} %')
                # print(f'time_if:{time_if}, percent:{(time_if/time_action_add)*100} %')
                # print(f'time_get_completion:{time_get_completion}, percent:{(time_get_completion/time_action_add)*100} %')
                # print(f'time_alter_frontier:{time_alter_frontier}, percent:{(time_alter_frontier/time_action_add)*100} %')
                # print(f'rest_time:{rest_time}, percent:{(rest_time/time_action_add)*100} %')
                # total_component_time = (time_if + time_expand_action + time_cal_lb + 
                #         time_get_completion + time_alter_frontier + 
                #         time_clip_time + time_get_head_state + 
                #         time_create_label_time + rest_time)

                # print(f'Sum of all components: {total_component_time}, percent: {(total_component_time/time_action_add)*100:.2f}%')
                # print('check for each action expand while loop')
                if did_gen_neg_red_cost==True:
                    #print('found complete path with this dual')
                    break
            #print('num_expanded_this_round')
            #print(num_expanded_this_round)
            if use_completion_on and did_gen_possible_expansion==False:
                print('********')
                print('********')
                print('********')
                print('********')
                print('curr_label.red_cost')
                print(curr_label.red_cost)
                print('curr_label.LB')
                print(curr_label.lb)
                print('len(curr_label.my_states_ordered)')
                print(len(curr_label.my_states_ordered))
                print('curr_label.all_nodes_ordered')
                print(curr_label.all_nodes_ordered)
                print('self.forbidden_nodes')
                print(self.forbidden_nodes)
                print('len(self.expandable_labels)')
                print(len(self.expandable_labels))
                can_complete=self.jy_get_compelition(new_label)
                print('can_complete')
                print(can_complete)
                input('big error here:use_completion_on and did_gen_possible_expansion==False')


        #print('DOEN T the CG process')
        #print('lowest_so_far')
        #print(lowest_so_far)
        #input('----')time_expand_action
        # print(len(all_routes))
        # print('route_gen_count')
        # print(route_gen_count)
        return all_routes
    
    def _get_dual_index_for_customer(self, customer):
        """
        Map customer ID to the corresponding index in the dual vector.
        
        Args:
            customer: Customer ID
            
        Returns:
            Index in the dual vector
        """
        # This is a placeholder - you would need to implement the specific mapping
        # between customer IDs and indices in your dual vector
        return customer  # Simplified assumption

    def _remove_labels_with_positive_lb(self):
        """
        Remove expandable labels with lower bound > 0.
        """
        new_expandable_labels = jy_sortedObject_list()
        debug_on=True
        for i, label in enumerate(self.expandable_labels.objects):
            if label.lb <= -0.0001:
                if debug_on==True:
                    tmp=set(label.all_nodes_ordered).intersection(self.forbidden_nodes)
                    if len(tmp)>0.5:
                        input('errorr')
                #my_tup=tuple([-len(label.my_states_ordered),label.red_cost,my_noise])
                new_tuple=self.label_2_tuple(label)

                new_expandable_labels.insert(label, new_tuple)
        
        self.expandable_labels = new_expandable_labels

    
    def pre_process__partition_actions(self):
        #call once prior to any optimization.  This partitions the actions for easy access
        
        self.actions_from_node=dict()
        #self.actions_from_node_subset_pickup=dict()
        self.actions_from_node_dest_dropoff=dict()
        #self.actions_from_node_subset_sink=dict()
        self.actions_from_node_MINUS_dest_dropoff=dict()
        for n in self.all_nodes:
            self.actions_from_node[n]=[]
            self.actions_from_node_MINUS_dest_dropoff[n]=[]
            self.actions_from_node_dest_dropoff[n]=[]
    
        for a in self.all_actions:
            my_origin=a.node_tail
            my_destination=a.node_head
            do_add=False
            if self.option_do_min_term:
                #shawn
                if np.min(self.node_min_term_vec[a.node_tail].toarray()- a.min_resource_vec.toarray())>=-0.0001:
                    do_add=True
                    self.actions_from_node[my_origin].append(a)
            else:
                do_add=True
            if do_add==True:
    
                self.actions_from_node[my_origin].append(a)
                if my_destination in self.dropoff_node:
                    self.actions_from_node_dest_dropoff[my_origin].append(a)
    
                else:
                    self.actions_from_node_MINUS_dest_dropoff[my_origin].append(a)
        
    def get_actions_from_label_2(self,my_label:jy_label,preferred_actions):

        if my_label.node == -1:
            actions_use = []
            for n in self.pickup_node:
                actions_use.append(self.action_dict[(my_label.node,n)][0])
            return actions_use
        if len(my_label.must_drop_off) ==0:
            actions_use = []
            actions_use.append(self.action_dict[(my_label.node,-2)][0])
            return actions_use
        if len(my_label.nodes_picked_up) >= self.jy_opt['max_pickups_in_a_route']:
            actions_use = []
            for n2 in my_label.must_drop_off:
                if n2+len(self.pickup_node) in preferred_actions[my_label.node]:
                    actions_use.append(self.action_dict[(my_label.node,n2+len(self.pickup_node))][0])
            return actions_use
        actions_use = set()

        # Part 1: Actions for drop-offs
        drop_off_nodes = [n+len(self.pickup_node) for n in my_label.must_drop_off]
        actions_use.update(self.action_dict[(my_label.node, node)][0] for node in drop_off_nodes if node in preferred_actions[my_label.node])

        # Get all relevant neighbors (from drop-offs and current node)
        #neighbor for must drop off (drop off)
        must_drop_off_neighbors_drop_off_node = set().union(*([this_node for this_node in self.neighbors[node] if this_node in preferred_actions[my_label.node]]
                                                                    for node in drop_off_nodes))
        #neighbor for current node
        neighbor_for_this_node = set(self.neighbors[my_label.node])
        #neighbor for must drop off (pick up)
        k= self.jy_opt['k_benefit_group']
        must_drop_off_neighbors_pick_up_node = set().union(*([this_node for this_node in list(self.benefit_group[node].keys() )[:k] if this_node in preferred_actions[my_label.node]] for node in my_label.must_drop_off))
        
        all_neighbors_to_check = must_drop_off_neighbors_drop_off_node | neighbor_for_this_node | must_drop_off_neighbors_pick_up_node
        # Find all valid pickup nodes at once
        valid_pickups = all_neighbors_to_check.intersection(self.pickup_node).difference(my_label.nodes_picked_up)
        actions_use.update(self.action_dict[(my_label.node, node)][0] for node in valid_pickups)
        return list(actions_use)
    def get_actions_from_label(self,my_label:jy_label):
        
        
        s=my_label.my_states_ordered[-1]
        if len(my_label.nodes_picked_up) < self.jy_opt['max_pickups_in_a_route']:
            actions_use=self.actions_from_node_MINUS_dest_dropoff[s.node].copy()
        else:
            actions_use = []
        must_dropoff=self.get_must_drop_off_including_current(s)
    
        if s.node!=-1 and len(must_dropoff)==0:
            actions_use=[]
            act_to_sink=self.action_dict[s.node,-2][0]
            actions_use=[act_to_sink]
        for n in must_dropoff:
            my_act_list=self.action_dict[(s.node,n)]
            for my_act in my_act_list:
                actions_use.append(my_act)
        return actions_use
    
    def get_must_drop_off_including_current(self,s):
        drop_off_node_need_to_visit=[]
        if s.node in self.pickup_node:
            #print('part one ')
            drop_off_node_need_to_visit = []
            drop_off_vec = s.state_vec[0,4+len(self.pickup_node):]
            dense_array = drop_off_vec.toarray()[0]
            zero_indices = np.where(dense_array == 0)[0]
            for n in zero_indices:
                drop_off_node_need_to_visit.append(n+1+len(self.dropoff_node))
            drop_off_node_need_to_visit.append(s.node+len(self.dropoff_node))
        elif s.node in self.dropoff_node:
            #print('part two ')

            drop_off_node_need_to_visit = []
            drop_off_vec = s.state_vec[0,4+len(self.pickup_node):]
            dense_array = drop_off_vec.toarray()[0]
            zero_indices = np.where(dense_array == 0)[0]
            for n in zero_indices:
                if n+1+len(self.pickup_node) != s.node:
                    drop_off_node_need_to_visit.append(n+1+len(self.dropoff_node))
        
        # may_avoid_dropoff=s.state_vec[4+self.num_pickups:]
        # must_dropoff=np.nonzero(may_avoid_dropoff<0.5)
        # must_dropoff=may_avoid_dropoff_list+self.num_pickups
        # if s.node in Q.pickup_node:
        #     must_dropoff.append(s.node+self.num_pickups)
        return drop_off_node_need_to_visit
    
    def jy_get_compelition(self,candid_label):
        
        if len(candid_label.all_nodes_ordered)==1:
            return True
        drop_off_needed=self.get_must_drop_off_including_current(candid_label.my_states_ordered[-1])
        if len(drop_off_needed)==0:
            return True
        #drop_off_needed=candid_label.node_wait_to_drop_off
        if len(drop_off_needed)>self.jy_opt['max_pickups_in_a_route']:
            return False
        permutation_of_drop_off_node = list(permutations(drop_off_needed))
        found_good_perm=False

        for my_perm_in in permutation_of_drop_off_node: 
            my_action_list_given_perm=[]
            
            my_perm=list(my_perm_in)
            #print('candid_label.all_nodes_ordered')
            #print(candid_label.all_nodes_ordered)
            #print('my_perm_in')
            #print(my_perm_in)
            my_perm=candid_label.all_nodes_ordered+my_perm+[-2]
            #print('my_perm')
            #print(my_perm)
            list_states=[candid_label.my_states_ordered[0]]
            is_good_perm=True

            for i in range(0,len(my_perm)-1):# in my_perm:
                pre_node = my_perm[i]
                post_node = my_perm[i+1]
                my_action=self.action_dict[(pre_node,post_node)][0]
                next_state=[]
                if self.jy_opt['use_load_ai_fast']==False:
                    next_state=my_action.get_head_state(list_states[-1],list_states[-1].l_id)
                else:
                    next_state=my_action.get_head_state_fast_load_ai(list_states[-1],list_states[-1].l_id)

                if next_state==None:
                    is_good_perm=False
                    break
                list_states.append(next_state)
                my_action_list_given_perm.append(my_action)
            if is_good_perm==True:
               # my_dedug_completion=jy_pricing_debug_completion(candid_state,orig_depth,list_depths,list_states,my_action_list_given_perm,True)
                found_good_perm=True
                #print('GOOD PERM IS')
                #print(my_perm)
                #print('----')
                break
        #if found_good_perm==False:
            #my_dedug_completion=jy_pricing_debug_completion(candid_state,orig_depth,None,None,None,False)
            #self.state_2_completion[candid_state]=my_dedug_completion
        
        return found_good_perm#self.state_2_completion[candid_state]
       