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

class jy_fast_pricing():
    """
    Class implementing a label-setting algorithm for finding minimum reduced cost paths
    with a given dual value.
    """

    def __init__(self, all_actions, action_dict, dual_vec, init_res_state, max_actions_in_route, actions_of_node, all_nodes, jy_opt):
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
        self.forbidden_nodes=[]

        self.dual_vec = dual_vec.copy()
        self.dual_vec_orig = dual_vec.copy()
        self.init_res_state = init_res_state
        self.max_actions_in_route = max_actions_in_route
        self.actions_of_node = actions_of_node
        self.all_nodes = all_nodes
        self.jy_opt = jy_opt
        self.num_cus = int((len(self.all_nodes)-2)/3)
        self.pickup_node = self.all_nodes[1:self.num_cus+1]
        self.dropoff_node = self.all_nodes[self.num_cus+1:2*self.num_cus+1]
        
        # Initialize additional attributes needed for the algorithm
        self.all_routes = []
        self.expandable_labels = jy_sortedObject_list()
        self.efficient_frontier = jy_efficient_frontier(all_nodes)
        
        # Prepare action reduced costs and find lowest reduced cost
        self.action_2_red_cost_dict = {}
        self.option_do_min_term = False
        self.lowest_action_contrib_red_cost = float('inf')
        self.pre_process__partition_actions()
        #print('initialization')

    def label_2_tuple(self,my_lab):
        
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
        self.all_routes = []
        self.expandable_labels = jy_sortedObject_list()
        self.efficient_frontier = jy_efficient_frontier(self.all_nodes)
        
        # Find minimum reduced cost paths
        routes = self.find_min_reduced_cost_path()
        
        # Return the routes found
        return routes

    def _compute_action_reduced_costs(self):
        """
        Compute reduced costs for all actions and identify the lowest reduced cost.
        """
        #forbidden_nodes=[]
        print('dual_vec')
        print(self.dual_vec)
        #print('type(self.dual_vec)')
        #print(type(self.dual_vec))
        ignore_vals=np.nonzero(np.array(self.dual_vec)<0.0001)[0]
        if len(ignore_vals)>0:
            pickup_forget=ignore_vals+1
            dropoff_forget=len(self.pickup_node)+pickup_forget
            dropoff_forget=list(dropoff_forget)
            pickup_forget=list(pickup_forget)
            self.forbidden_nodes=self.forbidden_nodes+pickup_forget+dropoff_forget
            self.forbidden_nodes=list(set(self.forbidden_nodes))
            #print('pickup_forget')
            #print(pickup_forget)
            #print('dropoff_forget')
            #print(dropoff_forget)
            print('self.forbidden_nodes')
            print(self.forbidden_nodes)
            #print('type(pickup_forget)')
            #print(type(pickup_forget))
            #print('type(dropoff_forget)')
            #print(type(dropoff_forget))
            #input('all_forget')
        bigVal=999999999999
        for action in self.all_actions:
            red_cost = action.comp_red_cost(self.dual_vec)
            if action.node_head in self.forbidden_nodes or action.node_tail in self.forbidden_nodes:
                red_cost=bigVal
                #input('hihihi')
            self.action_2_red_cost_dict[action] = red_cost
            
            if red_cost < self.lowest_action_contrib_red_cost:
                self.lowest_action_contrib_red_cost = red_cost
            #if action.node_head in self.forbidden_nodes or action.node_tail in self.forbidden_nodes:
            #    self.lowest_action_contrib_red_cost=bigVal
        #print('self.action_2_red_cost_dict')
        #print(self.action_2_red_cost_dict)
        #input('hi')
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
            action_2_red_cost_dict=self.action_2_red_cost_dict,
            actions_of_node=self.actions_of_node,
            jy_opt=self.jy_opt,
            pickup_nodes=self.pickup_node,
            dropoff_nodes=self.dropoff_node
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
            my_label.calculate_red_cost_given_dual(self.dual_vec,self.lowest_action_contrib_red_cost)
            tmp=set(my_label.all_nodes_ordered).intersection(set(self.forbidden_nodes))
            if len(tmp)>0.5:
                my_label.red_cost=np.inf
                my_label.lb=np.inf
            #if debug_on==True:
            #    if len(tmp)>0.5 and my_label.red_cost<100:
            #        print('my_label.red_cost')
            #        print(my_label.red_cost)
                  #  print('tmp')
                  #  print(tmp)
                  #  print('self.dual_vec')
                  ##  print(self.dual_vec)
                  #  input('error here')

    def find_min_reduced_cost_path(self):
        """
        Main method to find the minimum reduced cost path following the algorithm in the PDF.
        
        Returns:
            List of routes with negative reduced cost
        """
        # Initialize with source label
        self._compute_action_reduced_costs()
        source_label = self.initialize_source_label()
        self.all_routes = []
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
        while True:
            # Re-compute bounds based on dual values
            # Remove expandable labels with LB > 0
            num_expansion_out=num_expansion_out+1
            #print('num_expansion_out,num_expansion_in')
            #print(num_expansion_out,num_expansion_in)
            #input('redoing labels')
            self.update_red_cost_and_lb()
            self._remove_labels_with_positive_lb()

            # Update efficient frontier with current set of expandable labels
            for label in self.expandable_labels.objects:
                self.efficient_frontier.alter_fronteir_given_new_element(label)
            
            # Check if we can terminate
            min_lb = float('inf')
            for label in self.expandable_labels.objects:
                if label.lb < min_lb:
                    min_lb = label.lb
            
            if min_lb >= 0 or len(self.expandable_labels) == 0:
                break
            
            # Inner loop to process expandable labels
            #input('starting inner')
            num_expanded_this_round=0
            while len(self.expandable_labels) > 0:
                num_expansion_in=num_expansion_in+1
                num_expanded_this_round=num_expanded_this_round+1
                #print('num_expansion_out,num_expansion_in')
                #print([num_expansion_out,num_expansion_in])
                # Pop label with minimum current reduced cost
                curr_label = self.expandable_labels.pop()
                #print('curr_label.red_cost')
                #print(curr_label.red_cost)
                #print('curr_label.LB')
                #print(curr_label.lb)
                #print('len(curr_label.my_states_ordered)')
                #print(len(curr_label.my_states_ordered))
                #print('curr_label.all_nodes_ordered')
                #print(curr_label.all_nodes_ordered)
                #print('self.forbidden_nodes')
                #print(self.forbidden_nodes)
                # Generate all possible expansions for this label
                #expanded_labels = curr_label.expand_label_fully()
                poss_actions  = self.get_actions_from_label(curr_label)
                did_gen_neg_red_cost=False

                # Process each expanded label
                for my_act in poss_actions:
                    new_label = curr_label.expand_given_action(my_act)
                    if new_label == None:
                        continue
                    if new_label.lb>0:
                        continue
                    self.efficient_frontier.alter_fronteir_given_new_element(new_label)
                    if new_label.is_complete_route and new_label.red_cost < new_label.lb/10:
                        route = new_label.convert_2_route()
                        self.all_routes.append(route)
                        print('route made')
                        print('new_label.all_nodes_ordered')
                        print(new_label.all_nodes_ordered)
                        print('new_label.all_nodes_ordered')
                        print('new_label.red_cost')
                        print(new_label.red_cost)

                        self.dual_vec = self.dual_vec - route.Exog_vec*self.dual_vec_orig*alpha
                        
                        did_gen_neg_red_cost=True


                    elif not new_label.is_complete_route:
                        
                        new_tuple=self.label_2_tuple(new_label)

                        self.expandable_labels.insert(new_label,new_tuple )
                if did_gen_neg_red_cost==True:
                    break
            print('num_expanded_this_round')
            print(num_expanded_this_round)
        
        #print('DOEN T the CG process')
        #input('----')
        return self.all_routes

    
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
                    tmp=set(label.all_nodes_ordered).intersection(set(self.forbidden_nodes))
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
        
    
    def get_actions_from_label(self,my_label):
        s=my_label.my_states_ordered[-1]
        actions_use=self.actions_from_node_MINUS_dest_dropoff[s.node].copy()
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