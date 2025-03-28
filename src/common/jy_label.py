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
    def __init__(self,my_actions_ordered,my_states_ordered,red_cost,cost,parent_label,
                 dual_vec,max_actions_in_route,lowest_action_contrib_red_cost,
                 action_2_red_cost_dict,actions_of_node,action_dict,jy_opt,pickup_nodes,dropoff_nodes,
                 rcp_u_partial, rcp_d_partial,rcp_u_partial_2):
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
        self.must_drop_off = []
        self.rcp_u_partial = rcp_u_partial
        self.rcp_d_partial = rcp_d_partial
        self.rcp_u_partial_2 = rcp_u_partial_2
        
        for s in self.my_states_ordered:
            self.all_nodes_ordered.append(s.node)
            if s.node  in self.pickup_nodes:
                self.num_pickups_in_route=self.num_pickups_in_route+1
                self.nodes_picked_up.append(s.node)
                self.must_drop_off.append(s.node)
            if s.node  in self.dropoff_nodes:
                self.num_dropoffs_in_route=self.num_dropoffs_in_route+1
                self.nodes_dropped_off.append(s.node)
                self.must_drop_off.remove(s.node-len(self.pickup_nodes))
        self.num_dropoffs_needed=self.num_pickups_in_route-self.num_dropoffs_in_route
            #if self.jy_opt['using_load_ai_lazy'] and s.node>=-0.5 and s.node<=self.jy_opt['using_load_ai_lazy_num_pickups']:
            #    self.jy_num_pickups=self.jy_num_pickups+1
        #print('self.all_nodes_ordered')
        #print(self.all_nodes_ordered)
        #print('self.node_wait_to_drop_off')
        #print(self.node_wait_to_drop_off)
        #self.DEBUG_check_label_correct()
    def calculate_red_cost_given_dual(self,dual):
        red_cost =0
        for a in self.my_actions_ordered:
            red_cost += (a.cost - a.Exog_vec @ dual)
        self.red_cost = red_cost
        
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
            drop_off_d = d + len(self.pickup_nodes)
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
                D.append(self.node-len(self.pickup_nodes))

            tot_benefit_dropoff_dual = 0
            tot_benefit_droppoff_cost =0
            for d in D:
                drop_off_of_d = d +len(self.pickup_nodes)
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
                    tot_benefit_dropoff_pickup_cost += self.action_dict[(node,node+len(self.pickup_nodes))][0].cost
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
    def calculate_better_lb_2(self,dual):
        self.calculate_red_cost_given_dual(dual)
        
        if self.node == -1:
            self.lb =  -np.inf
            return self.lb
        elif self.node == -2:
            self.lb = self.red_cost
            return self.lb
        else:
            self.tot_gain = defaultdict()
            node_not_picked_up = list(set(self.pickup_nodes) - set(self.nodes_picked_up))
            
            D = self.must_drop_off[:]
            if self.node in self.dropoff_nodes:
                D.append(self.node-len(self.pickup_nodes))
            tot_benefit_dropoff_dual = 0
            tot_benefit_droppoff_cost =0
            for d in D:
                drop_off_of_d = d +len(self.pickup_nodes)
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
            
            sorted_node_with_k = defaultdict()
            for k in range(0,extra_customer_can_pick_up):
                tot_gain = defaultdict()
                for u in node_not_picked_up:
                    tot_gain[u] = -dual[u-1] + self.rcp_u_partial_2[(1+k+self.num_pickups_in_route,u)]#self.action_dict[(u,u+len(self.pickup_nodes))][0].cost/(1+k+self.num_pickups_in_route)
                tot_gain = dict(sorted(tot_gain.items(), key=lambda item: item[1]))
                sorted_node_with_k[k] = tot_gain

            for k in range(0,extra_customer_can_pick_up):
                sorted_key = list(sorted_node_with_k[k].keys())
                tot_benefit_dropoff_pickup_dual = 0
                tot_benefit_dropoff_pickup_cost = 0
                myDenom = k + self.num_pickups_in_route+1
                for node in sorted_key[:k+1]:
                    tot_benefit_dropoff_pickup_dual -= dual[node-1]
                    tot_benefit_dropoff_pickup_cost += self.action_dict[(node,node+len(self.pickup_nodes))][0].cost
                this_red_cost = self.red_cost + tot_benefit_dropoff_dual + tot_benefit_dropoff_pickup_dual + (tot_benefit_droppoff_cost+tot_benefit_dropoff_pickup_cost)/myDenom
                # myDenom = k + self.num_pickups_in_route+1
                # tot_benefit_dropoff_pickup_dual -= dual[sorted_key[k]-1]
                # tot_benefit_dropoff_pickup_cost += self.action_dict[(sorted_key[k],sorted_key[k]+len(self.pickup_nodes))][0].cost
                # this_red_cost = self.red_cost + tot_benefit_dropoff_dual + tot_benefit_dropoff_pickup_dual + (tot_benefit_droppoff_cost+tot_benefit_dropoff_pickup_cost)/myDenom
                if this_red_cost < lowest_red_cost:
                    lowest_red_cost = this_red_cost
            
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
            new_label=self.expand_given_action(my_act)
            if new_label!=None:
                all_labels_out.append(new_label)

        return all_labels_out
    def expand_given_action(self,my_action,dual_vec):
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

        if new_head!=None:
            NEW_my_actions_ordered=self.my_actions_ordered+[my_action]
            NEW_my_states_ordered=self.my_states_ordered+[new_head]
            NEW_red_cost=self.red_cost+self.action_2_red_cost_dict[my_action]
            NEW_cost=self.cost+my_action.cost
            NEW_parent_label=self
            NEW_label=jy_label(NEW_my_actions_ordered,NEW_my_states_ordered,NEW_red_cost,NEW_cost,NEW_parent_label,self.dual_vec,self.max_actions_in_route,self.lowest_action_contrib_red_cost,self.action_2_red_cost_dict,self.actions_of_node,self.action_dict,self.jy_opt,self.pickup_nodes,self.dropoff_nodes,self.rcp_u_partial,self.rcp_d_partial,self.rcp_u_partial_2)
            if self.jy_opt['lb_option'] == 2:
                NEW_label.calculate_better_lb_2(dual_vec)
            elif self.jy_opt['lb_option'] == 1:
                NEW_label.calculate_better_lb(dual_vec)
            elif self.jy_opt['lb_option'] == 0:
                NEW_label.calculate_lb_given_lowest_action_contrib_red_cost(self.lowest_action_contrib_red_cost)
            elif self.jy_opt['lb_option'] == 'check':
                lb1 = NEW_label.calculate_better_lb(dual_vec)
                lb2 = NEW_label.calculate_better_lb_2(dual_vec)
                if lb1<lb2:
                    input('lb error here')
            else:
                input('no lb option used')
            if self.lb>NEW_label.lb+.001:
                print('self.lb')
                print(self.lb)
                print('NEW_label.lb')
                print(NEW_label.lb)
                print('self.all_nodes_ordered')
                print(self.all_nodes_ordered)
                print('NEW_label.all_nodes_ordered')
                print(NEW_label.all_nodes_ordered)
                print('self.red_cost')
                print(self.red_cost)
                print('NEW_label.red_cost')
                print(NEW_label.red_cost)
                print('dual_vec[0]')
                print(dual_vec[0])
                print('self.my_actions_ordered[1].cost')
                print(self.my_actions_ordered[1].cost)
                print('gap is ')
                print('NEW_label.lb-self.lb')
                print(NEW_label.lb-self.lb)
                offset_pickup=1
                offset_dropoff=6
                print('dual_vec[4-1]')

                print(dual_vec[4-offset_pickup])
                print('dual_vec[2-1]')
                print(dual_vec[2-offset_pickup])
                print('dual_vec[9-1]')
                print(dual_vec[9-offset_dropoff])
                print('dual_vec[7-1]')
                print(dual_vec[7-offset_dropoff])
                print('self.action_dict[4,2][0].cost')
                print(self.action_dict[4,2][0].cost)
                print('self.action_dict[2,9][0].cost')
                print(self.action_dict[2,9][0].cost)
                print('self.action_dict[9,7][0].cost')
                print(self.action_dict[9,7][0].cost)
                input('error here the lb went down')
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
