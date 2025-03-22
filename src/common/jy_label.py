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

class jy_label:
#     
    def __init__(self,my_actions_ordered,my_states_ordered,red_cost,cost,parent_label,dual_vec,max_actions_in_route,lowest_action_contrib_red_cost,action_2_red_cost_dict,actions_of_node,action_dict,jy_opt,pickup_nodes,dropoff_nodes):
        self.jy_opt=jy_opt
        self.my_actions_ordered=my_actions_ordered
        self.my_states_ordered=my_states_ordered
        self.parent_label=parent_label
        self.red_cost=red_cost
        self.cost=cost
        self.dual_vec=dual_vec
        self.node=self.my_states_ordered[-1].node
        self.max_actions_in_route=max_actions_in_route
        self.lowest_action_contrib_red_cost=lowest_action_contrib_red_cost
        self.action_2_red_cost_dict=action_2_red_cost_dict
        self.actions_of_node=actions_of_node
        self.action_dict = action_dict
        self.lb=self.red_cost+(max_actions_in_route-len(self.my_actions_ordered))*lowest_action_contrib_red_cost
        self.pickup_nodes=pickup_nodes
        self.dropoff_nodes=dropoff_nodes

        self.is_complete_route=self.my_states_ordered[-1].is_sink
        self.all_nodes_ordered=[]
        self.num_dropoffs_in_route=0
        self.num_pickups_in_route=0
        self.nodes_picked_up = []
        self.nodes_dropped_off = []
        self.node_wait_to_drop_off = []
        for s in self.my_states_ordered:
            self.all_nodes_ordered.append(s.node)
            if s.node  in self.pickup_nodes:
                self.num_pickups_in_route=self.num_pickups_in_route+1
                self.nodes_picked_up.append(s.node)
                self.node_wait_to_drop_off.append(s.node)
            if s.node  in self.dropoff_nodes:
                self.num_dropoffs_in_route=self.num_dropoffs_in_route+1
                self.nodes_dropped_off.append(s.node)
                self.node_wait_to_drop_off.remove(s.node-len(self.pickup_nodes))
        self.num_dropoffs_needed=self.num_pickups_in_route-self.num_dropoffs_in_route
            #if self.jy_opt['using_load_ai_lazy'] and s.node>=-0.5 and s.node<=self.jy_opt['using_load_ai_lazy_num_pickups']:
            #    self.jy_num_pickups=self.jy_num_pickups+1
        
        self.DEBUG_check_label_correct()
    def calculate_red_cost_given_dual(self,dual):
        red_cost =0
        for a in self.my_actions_ordered:
            red_cost += (a.cost - a.Exog_vec @ dual)
        self.red_cost = red_cost
        
    def calculate_lb_given_lowest_action_contrib_red_cost(self,lowest_action_contrib_red_cost):
        self.lb = self.red_cost+(self.max_actions_in_route-len(self.my_actions_ordered))*lowest_action_contrib_red_cost
    def calculate_better_lb(self, dual):
        q = self.jy_opt['max_pickups_in_a_route'] - self.num_pickups_in_route
        D = self.node_wait_to_drop_off
        V = self.jy_opt['max_pickups_in_a_route'] - self.num_pickups_in_route
        F = list(set(self.pickup_nodes) - set(self.nodes_picked_up))
        def get_base_gain():
            base_gain =0
            for d in D:
                
                coef =1
                if d-len(self.pickup_nodes) == self.node:
                    coef +=1
                pi_d = self.dual_vec[0,d-len(self.pickup_nodes)-1]
                this_gain = pi_d*coef*-1/2 + self.action_dict[(self.node,d)].cost
                base_gain += this_gain
            return base_gain

        def get_tot_gain(k_minus_D):
            tot_tain = 0
            poss_drop_off = list(combinations(F,k_minus_D))
            for drop_off_nodes in poss_drop_off:
                for f in drop_off_nodes:
                    gain_f = self.action_dict[(f,f+len(self.pickup_nodes ))]
                    tot_tain += gain_f
            return tot_tain
        
        base_gain = get_base_gain()
        min_lb = np.inf
        for k in range(len(D),len(D)+V):
            tot_gain = get_tot_gain(k-len(D))
            this_lb = (base_gain+tot_gain)/(min(k,V))
            if this_lb < min_lb:
                min_lb = this_lb
        LB = self.calculate_red_cost_given_dual(dual) + min_lb
        self.lb =  LB
        
    def this_label_dominates_input(self,candid_label):
        
        my_flag=True
        is_identical=True
        if self.red_cost>candid_label.red_cost:
            my_flag=False
        if self.lb>candid_label.lb:
            my_flag=False
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
            new_label=self.expand_given_action(my_act)
            if new_label!=None:
                all_labels_out.append(new_label)

        return all_labels_out
    def expand_given_action(self,my_action):
        NEW_label=None
        last_state=self.my_states_ordered[-1]
        new_head=my_action.get_head_state(last_state,last_state.l_id)

        if self.max_actions_in_route<len(self.my_actions_ordered) :
            input('errror here not posible')
        if self.max_actions_in_route==len(self.my_actions_ordered) and my_action.node_head!=-2:
            input('errror here not posible 2')

        if new_head!=None:
            NEW_my_actions_ordered=self.my_actions_ordered+[my_action]
            NEW_my_states_ordered=self.my_states_ordered+[new_head]
            NEW_red_cost=self.red_cost+self.action_2_red_cost_dict[my_action]
            NEW_cost=self.cost+my_action.cost
            NEW_parent_label=self
            NEW_label=jy_label(NEW_my_actions_ordered,NEW_my_states_ordered,NEW_red_cost,NEW_cost,NEW_parent_label,self.dual_vec,self.max_actions_in_route,self.lowest_action_contrib_red_cost,self.action_2_red_cost_dict,self.actions_of_node,self.jy_opt,self.pickup_nodes,self.dropoff_nodes)
        return NEW_label
    
    def convert_2_route(self):
        if self.my_states_ordered[-1].node!=-2:
            input('this route is not done')
            input('---')
        state_action_alt_repeat=[self.my_states_ordered[0]]
        for i in range(0,len(self.my_actions_ordered)):
            state_action_alt_repeat.append(self.my_actions_ordered[i])
            state_action_alt_repeat.append(self.my_states_ordered[i+1])
        my_route=Route(state_action_alt_repeat,0)
        
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
