import bisect
import numpy as np

from numpy import zeros, ones
from src.common.helper import Helper
from scipy.sparse import csr_matrix
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from src.common.state import State
from src.common.action import Action
import numpy as np
from src.common.pgm_approach import Route
from itertools import combinations
import time
class jy_label:
#     
    def __init__(self,my_actions_ordered,my_states_ordered,red_cost,cost,parent_label:'jy_label',
                 dual_vec,max_actions_in_route,lowest_action_contrib_red_cost,
                 actions_of_node,action_dict,jy_opt,cus_num,pickup_nodes,dropoff_nodes,rcp_u_partial_2,edges,preferred_actions, distance):
        self.jy_opt=jy_opt
        self.my_actions_ordered=my_actions_ordered
        if not my_actions_ordered:
        # Handle empty list case
            self.my_actions_ordered = []
            self.red_cost_non_zero_cal_indices = np.array([], dtype=int)
            self.red_cost_non_zero_cal_vals = np.array([])
            self.total_cost = 0
        else:
            last_action = my_actions_ordered[-1]
            self.total_cost = parent_label.total_cost+last_action.cost
            Exog_vec_non_zero_indices = last_action.red_cost_non_zero_cal_indices
            Exog_vec_non_zero_val = last_action.red_cost_non_zero_cal_vals
            
            self.red_cost_non_zero_cal_indices = list(parent_label.red_cost_non_zero_cal_indices)
            self.red_cost_non_zero_cal_vals = list(parent_label.red_cost_non_zero_cal_vals)
            if Exog_vec_non_zero_indices is not None:
                if Exog_vec_non_zero_indices not in self.red_cost_non_zero_cal_indices:
                    self.red_cost_non_zero_cal_indices.append(Exog_vec_non_zero_indices)
                    self.red_cost_non_zero_cal_vals.append(Exog_vec_non_zero_val)
                else:
                    index = self.red_cost_non_zero_cal_indices.index(Exog_vec_non_zero_indices)
                    self.red_cost_non_zero_cal_vals[index] += Exog_vec_non_zero_val

        self.my_states_ordered=my_states_ordered
        self.parent_label=parent_label
        self.red_cost=red_cost
        self.cost=cost
        self.dual_vec=dual_vec
        self.cus_num = cus_num
        self.node=self.my_states_ordered[-1].node
        self.max_actions_in_route=max_actions_in_route
        self.lowest_action_contrib_red_cost=lowest_action_contrib_red_cost
        #self.action_2_red_cost_dict=action_2_red_cost_dict
        self.actions_of_node=actions_of_node
        self.action_dict = action_dict
        self.lb=self.red_cost+(max_actions_in_route-len(self.my_actions_ordered))*lowest_action_contrib_red_cost
        self.pickup_nodes=pickup_nodes
        self.dropoff_nodes=dropoff_nodes

        self.is_complete_route=self.my_states_ordered[-1].is_sink
        if parent_label == None:
            self.all_nodes_ordered = [-1]
            self.nodes_picked_up = set()
            self.nodes_dropped_off = set()
            self.must_drop_off = set()
        else:
            self.all_nodes_ordered=parent_label.all_nodes_ordered + [self.node]

            
            self.nodes_picked_up = set(parent_label.nodes_picked_up)
            self.nodes_dropped_off = set(parent_label.nodes_dropped_off)
            self.must_drop_off = set(parent_label.must_drop_off)
            if self.node <= self.cus_num:
                self.nodes_picked_up.add(self.node)
                self.must_drop_off.add(self.node)
            elif self.node != -2:
                self.nodes_dropped_off.add(self.node - self.cus_num)
                self.must_drop_off.remove(self.node - self.cus_num)

        self.num_dropoffs_in_route= len(self.nodes_dropped_off)
        self.num_pickups_in_route=len(self.nodes_picked_up)
        self.num_dropoffs_needed=len(self.must_drop_off)
        self.rcp_u_partial_2 = rcp_u_partial_2
        self.edges = edges
        self.preferred_actions = preferred_actions
        self.distance = distance

        # for s in self.my_states_ordered:
        #     self.all_nodes_ordered.append(s.node)
        #     if s.node  in self.pickup_nodes:
        #         self.num_pickups_in_route=self.num_pickups_in_route+1
        #         self.nodes_picked_up.append(s.node)
        #         self.must_drop_off.append(s.node)
        #     if s.node  in self.dropoff_nodes:
        #         self.num_dropoffs_in_route=self.num_dropoffs_in_route+1
        #         self.nodes_dropped_off.append(s.node)
        #         self.must_drop_off.remove(s.node-len(self.pickup_nodes))
        
            #if self.jy_opt['using_load_ai_lazy'] and s.node>=-0.5 and s.node<=self.jy_opt['using_load_ai_lazy_num_pickups']:
            #    self.jy_num_pickups=self.jy_num_pickups+1
        #print('self.all_nodes_ordered')
        #print(self.all_nodes_ordered)
        #print('self.node_wait_to_drop_off')
        #print(self.node_wait_to_drop_off)
        #self.DEBUG_check_label_correct()
    def calculate_red_cost_given_dual(self,dual):
        if not self.my_actions_ordered:
            self.red_cost = 0
            return 0

    
        # Use pre-computed values and indices for dot product
        if len(self.red_cost_non_zero_cal_indices) > 0:
            dot_product = np.dot(self.red_cost_non_zero_cal_vals, 
                                dual[self.red_cost_non_zero_cal_indices])
        else:
            dot_product = 0
        
        self.red_cost = self.total_cost - dot_product
        return self.red_cost
        
    def calculate_lb_given_lowest_action_contrib_red_cost(self,lowest_action_contrib_red_cost):
        self.lb = self.red_cost+(self.max_actions_in_route-len(self.my_actions_ordered))*lowest_action_contrib_red_cost
    def calculate_rcp_with_dual(self,dual):
        self.base_gain = defaultdict()
        self.tot_gain = defaultdict()
        node_not_picked_up = list(set(self.pickup_nodes) - set(self.nodes_picked_up))
        #print('self.all_nodes')
        #print(self.all_nodes_ordered)
        #print('node_wait_to_drop_off')
        #print(self.node_wait_to_drop_off)
        #input('--')
        #if self.all_nodes_ordered == [-1, 4, 2, 9]:
                #print('check here')
        for d in self.must_drop_off:
            drop_off_d = d + self.cus_num
            if d == self.node:
               # print('in here')
               # print(d)
               # print('in here')
                self.base_gain[d] = -dual[d-1] + self.rcp_d_partial[(self.node,drop_off_d)]
            else:
                self.base_gain[d] = -dual[d-1]/2 + self.rcp_d_partial[(self.node,drop_off_d)]
            
            if drop_off_d == self.node:
                self.base_gain[d] = -dual[d-1]/2
        for u in node_not_picked_up:
            self.tot_gain[u] = -dual[u-1] + self.rcp_u_partial[u]
        self.tot_gain = dict(sorted(self.tot_gain.items(), key=lambda item: item[1]))
        #print('check here')
    
    def calculate_better_lb(self,dual):
        self.calculate_rcp_with_dual(dual)
        if self.node == -1:
            self.lb =  -np.inf
            return self.lb
        elif self.node == -2:
            self.lb = self.red_cost
            return self.lb
        else:
            D = self.must_drop_off
            V = self.jy_opt['max_pickups_in_a_route'] - self.num_pickups_in_route
            #F = list(set(self.pickup_nodes) - set(self.nodes_picked_up))
            D = self.must_drop_off[:]
            if self.node in self.dropoff_nodes:
                D.append(self.node-self.cus_num)

            tot_benefit_dropoff_dual = 0
            tot_benefit_droppoff_cost =0
            for d in D:
                drop_off_of_d = d +self.cus_num
                if drop_off_of_d == self.node:
                    tot_benefit_dropoff_dual -= dual[d-1]/2
                elif d == self.node:
                    tot_benefit_dropoff_dual -= dual[d-1]
                    tot_benefit_droppoff_cost += self.action_dict[(self.node,drop_off_of_d)][0].cost
                else:
                    tot_benefit_dropoff_dual -= dual[d-1]/2
                    tot_benefit_droppoff_cost += self.action_dict[(self.node,drop_off_of_d)][0].cost
            tot_benefit_dropoff_pickup_dual = 0
            tot_benefit_dropoff_pickup_cost = 0
            extra_customer_can_pick_up = self.jy_opt['max_pickups_in_a_route']-self.num_pickups_in_route
            # not picking new customer as first lowest red cost
            if self.num_pickups_in_route > 0.5:
                lowest_red_cost = self.red_cost+tot_benefit_dropoff_dual + tot_benefit_droppoff_cost/self.num_pickups_in_route
            else:
                lowest_red_cost = np.inf
            
            for k in range(0,extra_customer_can_pick_up):
                sorted_key = list(self.tot_gain.keys())
                tot_benefit_dropoff_pickup_dual = 0
                tot_benefit_dropoff_pickup_cost = 0
                myDenom = self.jy_opt['max_pickups_in_a_route']
                for node in sorted_key[:k+1]:
                    tot_benefit_dropoff_pickup_dual -= dual[node-1]
                    tot_benefit_dropoff_pickup_cost += self.action_dict[(node,node+self.cus_num)][0].cost
                this_red_cost = self.red_cost + tot_benefit_dropoff_dual + tot_benefit_dropoff_pickup_dual + (tot_benefit_droppoff_cost+tot_benefit_dropoff_pickup_cost)/myDenom
                # myDenom = k + self.num_pickups_in_route+1
                # tot_benefit_dropoff_pickup_dual -= dual[sorted_key[k]-1]
                # tot_benefit_dropoff_pickup_cost += self.action_dict[(sorted_key[k],sorted_key[k]+len(self.pickup_nodes))][0].cost
                # this_red_cost = self.red_cost + tot_benefit_dropoff_dual + tot_benefit_dropoff_pickup_dual + (tot_benefit_droppoff_cost+tot_benefit_dropoff_pickup_cost)/myDenom
                if this_red_cost < lowest_red_cost:
                    lowest_red_cost = this_red_cost
            self.lb = lowest_red_cost
            return self.lb
            #print('lb')
            #print(lb)
            #input('----')
    def calculate_better_lb_2(self, dual,sorted_node_with_k):
        self.calculate_red_cost_given_dual(dual)
        
        if self.node == -1:
            self.lb = -np.inf
            return self.lb
        elif self.node == -2:
            self.lb = self.red_cost
            return self.lb
        else:
            # Optimize by pre-computing sets
            #node_not_picked_up = list(pickup_nodes_set - nodes_picked_up_set)
            
            #D = list(self.must_drop_off.copy())  # Use .copy() instead of [:] for clarity
            
            # Check if node is in dropoff nodes once
            is_node_in_dropoff = self.node in self.dropoff_nodes
            pickup_nodes_len = self.cus_num
            
            if is_node_in_dropoff:
                this_must_drop_off = set(self.must_drop_off)
                this_must_drop_off.add(self.node - pickup_nodes_len)
                D = list(this_must_drop_off)
            else:
                D = list(self.must_drop_off)    
            #D.append(self.node - pickup_nodes_len)
            
            # Consolidate loops and calculations
            tot_benefit_dropoff_dual = 0
            tot_benefit_droppoff_cost = 0

            for d in D:
                drop_off_of_d = d + pickup_nodes_len
                if drop_off_of_d == self.node:
                    tot_benefit_dropoff_dual -= dual[d-1] / 2
                elif d == self.node:
                    tot_benefit_dropoff_dual -= dual[d-1]
                    #tot_benefit_droppoff_cost += self.action_dict[(self.node, drop_off_of_d)][0].cost
                    tot_benefit_droppoff_cost += self.distance[self.node, drop_off_of_d]
                else:
                    tot_benefit_dropoff_dual -= dual[d-1] / 2
                    #tot_benefit_droppoff_cost += self.action_dict[(self.node, drop_off_of_d)][0].cost
                    tot_benefit_droppoff_cost += self.distance[self.node, drop_off_of_d]
            # Pre-calculate constants
            extra_customer_can_pick_up = self.jy_opt['max_pickups_in_a_route'] - self.num_pickups_in_route
            
            # Initialize lowest_red_cost once
            if self.num_pickups_in_route > 0.5:
                lowest_red_cost = self.red_cost + tot_benefit_dropoff_dual + tot_benefit_droppoff_cost / self.num_pickups_in_route
            else:
                lowest_red_cost = float('inf')  # Use float('inf') instead of np.inf for better performance
            

            base_cost = self.red_cost + tot_benefit_dropoff_dual
            # Optimize the final loop to calculate the best lower bound
            for k in range(1,extra_customer_can_pick_up+1):
                myDenom = k + self.num_pickups_in_route
                sorted_key = list(sorted_node_with_k[myDenom].keys())
                
                # Skip unnecessary computation if there are no keys
                if not sorted_key:
                    continue
                    
                # Only take as many keys as are available or needed
                nodes_to_use = sorted_key[:k]
                
                # Only calculate if we have enough nodes
                if len(nodes_to_use) == k:
                    
                    benefit_sum = sum(sorted_node_with_k[myDenom][key] for key in nodes_to_use)
                    this_red_cost = base_cost + tot_benefit_droppoff_cost / myDenom + benefit_sum
                    

                    if this_red_cost < lowest_red_cost:
                        lowest_red_cost = this_red_cost
                else:
                    input('error here: if len(nodes_to_use) == k')
            self.lb = lowest_red_cost
            return self.lb
    def this_label_dominates_input(self,candid_label):
        
        my_flag=True
        is_identical=True
        if self.red_cost>candid_label.red_cost:
            my_flag=False
            return[my_flag, is_identical]
        if self.lb>candid_label.lb:
            my_flag=False
            return[my_flag, is_identical]
        state_does_dom, state_does_equal=self.my_states_ordered[-1].this_state_dominates_input_state(candid_label.my_states_ordered[-1])
        if state_does_dom==False and state_does_equal==False:
            my_flag=False
        
        is_identical=False
        if state_does_equal==True and self.red_cost==candid_label.red_cost and self.lb==candid_label.lb:
            is_identical=True
        if is_identical==True:
            my_flag=False
        return [my_flag,is_identical]
    
    def expand_label_fully(self):
        
        my_last_state=self.my_states_ordered[-1]
        my_node=my_last_state.node
        all_labels_out=[]
        for my_act in self.actions_of_node[my_node]:
            if self.jy_opt['using_load_ai_lazy']==True:
                #print('looking to expand ')
                #print('self.all_nodes_ordered')
                #print(self.all_nodes_ordered)
                #print('my_act.node_head')
                #print(my_act.node_head)
                #input('---')
                if my_act.node_head>2*self.jy_opt['using_load_ai_lazy_num_pickups']:
                    #print('kill 1')
                    #input('--killin g -')
                    continue
                if my_act.node_head>self.jy_opt['using_load_ai_lazy_num_pickups'] and my_act.node_head<=2*self.jy_opt['using_load_ai_lazy_num_pickups']:
                    cust_pickup=my_act.node_head-self.jy_opt['using_load_ai_lazy_num_pickups']
                    if cust_pickup not in self.all_nodes_ordered:
                        #print('kill 2')
                        #input('kill ing 2 ')
                        continue
                if my_act.node_head<=self.jy_opt['using_load_ai_lazy_num_pickups'] and my_act.node_head>-0.5 and self.jy_num_pickups==self.jy_opt['using_load_ai_lazy_max_pickups']:
                    #print('kill 3')
                    continue
                if my_act.node_head in self.all_nodes_ordered:
                    #print('kill 4')
                    continue
            #print('my_act.node_head')
            #print(my_act.node_head)
            #input('--')
            new_label, get_head_state_time, cal_lb_time=self.expand_given_action(my_act)
            if new_label!=None:
                all_labels_out.append(new_label)

        return all_labels_out
    def expand_given_action(self,my_action,dual,forbidden_nodes):

        NEW_label=None
        last_state=self.my_states_ordered[-1]
        new_head=[]

        if self.jy_opt['use_load_ai_fast']==False:
            new_head=my_action.get_head_state(last_state,last_state.l_id)
        else:
            new_head=my_action.get_head_state_fast_load_ai(last_state,last_state.l_id)
        if self.max_actions_in_route<len(self.my_actions_ordered) :
            input('errror here not posible')
        if self.max_actions_in_route==len(self.my_actions_ordered) and my_action.node_head!=-2:
            input('errror here not posible 2')
        bigVal=999999999999
        #print(f'time for get head state {time2-time1}')
        if new_head!=None:
            NEW_my_actions_ordered=self.my_actions_ordered+[my_action]
            NEW_my_states_ordered=self.my_states_ordered+[new_head]
            if my_action.node_head in forbidden_nodes or my_action.node_tail in forbidden_nodes:
                NEW_red_cost = bigVal
            else:
                NEW_red_cost = self.red_cost + my_action.comp_red_cost(dual)
            #NEW_red_cost=self.red_cost+self.action_2_red_cost_dict[my_action]
            NEW_cost=self.cost+my_action.cost
            NEW_parent_label=self
            NEW_label=jy_label(NEW_my_actions_ordered,NEW_my_states_ordered,NEW_red_cost,NEW_cost,NEW_parent_label,
                               self.dual_vec,self.max_actions_in_route,self.lowest_action_contrib_red_cost,
                               self.actions_of_node,self.action_dict,self.jy_opt,self.cus_num,self.pickup_nodes,
                               self.dropoff_nodes,self.rcp_u_partial_2,self.edges,self.preferred_actions,self.distance)
        return NEW_label
    
    def convert_2_route(self):
        if self.my_states_ordered[-1].node!=-2:
            input('this route is not done')
            input('---')
        state_action_alt_repeat=[self.my_states_ordered[0]]
        for i in range(0,len(self.my_actions_ordered)):
            state_action_alt_repeat.append(self.my_actions_ordered[i])
            state_action_alt_repeat.append(self.my_states_ordered[i+1])
        my_route=Route(state_action_alt_repeat,0,self.pickup_nodes)
        
        return my_route

    def DEBUG_check_label_correct(self):
        cur_cost=0
        cur_red_cost=0
        for ai in range(0,len(self.my_actions_ordered)):
            this_act=self.my_actions_ordered[ai]
            state_tail_this_act=self.my_states_ordered[ai]
            state_head_this_act=self.my_states_ordered[ai+1]
            new_head=this_act.get_head_state(state_tail_this_act,state_tail_this_act.l_id)
            is_equal=state_head_this_act.equals_minus_id(new_head)
            if is_equal==False:
                input('error here')
            cur_cost=cur_cost+this_act.cost
            cur_red_cost=cur_red_cost+this_act.comp_red_cost(self.dual_vec)
            
        if np.abs(cur_cost-self.cost)>.0001 or (cur_red_cost-self.red_cost)>.0001:
            input('errror here in cost')
