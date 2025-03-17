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

class SortedObjectList:
    """Maintains a sorted list of objects based on an associated scalar value."""
    
    def __init__(self):
        self.values = []  # List of scalar values (used for sorting)
        self.objects = []  # List of associated objects

    def insert(self, obj, value):
        """Inserts an object while keeping the list sorted by value."""
        index = bisect.bisect_left(self.values, value)  # Find insertion index
        self.values.insert(index, value)  # Insert value in sorted order
        self.objects.insert(index, obj)  # Insert object in corresponding position

    def pop(self):
        """Removes and returns the object with the smallest value."""
        if not self.objects:
            raise IndexError("Pop from empty SortedObjectList")
        self.values.pop(0)  # Remove first (smallest) value
        return self.objects.pop(0)  # Remove and return first object

    def pop_max(self):
        """Removes and returns the object with the largest value."""
        if not self.objects:
            raise IndexError("Pop from empty SortedObjectList")
        self.values.pop()  # Remove last (largest) value
        return self.objects.pop()  # Remove and return last object

    def __len__(self):
        return len(self.objects)

    def __repr__(self):
        return str(list(zip(self.values, self.objects)))  # Show sorted pairs


class label:

    def __init__(self,my_actions_ordered,my_states_ordered,red_cost,cost,parent_label,dual_vec,max_actions_in_route,lowest_action_contrib_red_cost,action_2_red_cost_dict,actions_of_node,jy_opt):
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
        self.lb=self.red_cost+(max_actions_in_route-len(self.my_actions_ordered))*lowest_action_contrib_red_cost


        self.is_complete_route=self.my_states_ordered[-1].is_sink
        self.all_nodes_ordered=[]
        self.jy_num_pickups=0
        for s in self.my_states_ordered:
            self.all_nodes_ordered.append(s.node)
            if self.jy_opt['using_load_ai_lazy'] and s.node>=-0.5 and s.node<self.jy_opt['using_load_ai_lazy_num_pickups']:
                self.jy_num_pickups=self.jy_num_pickups+1
        
        self.DEBUG_check_label_correct()
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
                if my_act.node_head>=2*self.jy_opt['using_load_ai_lazy_num_pickups']:
                    #print('kill 1')
                    #input('--killin g -')
                    continue
                if my_act.node_head>=self.jy_opt['using_load_ai_lazy_num_pickups'] and my_act.node_head<2*self.jy_opt['using_load_ai_lazy_num_pickups']:
                    cust_pickup=my_act.node_head-self.jy_opt['using_load_ai_lazy_num_pickups']
                    if cust_pickup not in self.all_nodes_ordered:
                        #print('kill 2')
                        #input('kill ing 2 ')
                        continue
                if my_act.node_head<self.jy_opt['using_load_ai_lazy_num_pickups'] and my_act.node_head>-0.5 and self.jy_num_pickups==self.jy_opt['using_load_ai_lazy_max_pickups']:
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
                #print('new_label.all_nodes_ordered')
                #print(new_label.all_nodes_ordered)
        #print('from ')
        #print(self.all_nodes_ordered)
        #input('---')
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
            NEW_label=label(NEW_my_actions_ordered,NEW_my_states_ordered,NEW_red_cost,NEW_cost,NEW_parent_label,self.dual_vec,self.max_actions_in_route,self.lowest_action_contrib_red_cost,self.action_2_red_cost_dict,self.actions_of_node,self.jy_opt)
        return NEW_label
    
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

class efficient_frontier:
    def __init__(self,all_nodes):
        self.all_nodes=all_nodes
        self.node_2_eff_fronteir=dict()
        for my_node in all_nodes:
            self.node_2_eff_fronteir[my_node]=set([])

    def is_in_fronteir(self,input_label):
        if input_label in self.node_2_eff_fronteir[input_label.node]:
            return True
        return False

    def alter_fronteir_given_new_element(self,new_label):

        is_in_frontier=True

        for old_label in self.node_2_eff_fronteir[new_label.node]:
            does_dom,does_equal = old_label.this_label_dominates_input(new_label)
            if does_dom==True or does_equal==True :
                is_in_frontier=False
                #print('old_label')
                #print(old_label.all_nodes_ordered)
                #print('new_label')
                #print(new_label.all_nodes_ordered)
                #input('--not adding --')
                break
        #print('is_in_frontier')
       # print(is_in_frontier)

        if is_in_frontier==True:
            labels_input_dominates=[]

            for old_label in self.node_2_eff_fronteir[new_label.node]:
                does_dom,does_equal=new_label.this_label_dominates_input(old_label)
                labels_input_dominates.append(old_label)
            
            for old_label in labels_input_dominates:#self.node_2_eff_fronteir[new_label.node]:
                self.node_2_eff_fronteir[new_label.node].remove(old_label)
            
            self.node_2_eff_fronteir[new_label.node].add(new_label)



class jy_slow_general_pricing_solver:

    def create_action_2_red_cost_dict(self):
        self.action_2_red_cost=dict()
        self.lowest_action_contrib_red_cost=np.inf
        for my_act in self.all_actions:
            this_red_cost=my_act.comp_red_cost(self.dual_vec)
            self.action_2_red_cost[my_act]=this_red_cost
            self.lowest_action_contrib_red_cost=np.min([self.lowest_action_contrib_red_cost,this_red_cost])
        #print('self.lowest_action_contrib_red_cost')
        #print(self.lowest_action_contrib_red_cost)
        #input('hold')
    def create_init_label(self):

        my_actions_ordered=[]
        my_states_ordered=[self.init_res_state]
        red_cost=0
        cost=0
        parent_label=None

        init_label=label(my_actions_ordered,my_states_ordered,red_cost,cost,parent_label,self.dual_vec,self.max_actions_in_route,self.lowest_action_contrib_red_cost,self.action_2_red_cost,self.actions_of_node,self.jy_opt)
        return init_label

    def compute_actions_of_each_node(self):
        self.actions_of_node=dict()
        for n in self.all_nodes:
            self.actions_of_node[n]=[]
        for a in self.all_actions:
            self.actions_of_node[a.node_tail].append(a)


    def add_label(self,my_label_add):

        if my_label_add.lb<0:
            self.my_efficient_frontier.alter_fronteir_given_new_element(my_label_add)
            if self.my_efficient_frontier.is_in_fronteir(my_label_add):
                my_coef=my_label_add.lb*(self.jy_opt['weight_expand_lb'])+my_label_add.red_cost*(1-self.jy_opt['weight_expand_lb'])
                self.unexpand_labels.insert(my_label_add,my_coef)

    def __init__(self,all_actions,dual_vec,init_res_state,max_actions_in_route,actions_of_node,all_nodes,jy_opt):
        self.all_nodes=all_nodes
        self.all_actions=all_actions
        self.jy_opt=jy_opt
        if 'min_red_cost_terminate_early' not in jy_opt:
            self.jy_opt['min_red_cost_terminate_early']=-1
        if 'weight_expand_lb' not in jy_opt:
            self.jy_opt['weight_expand_lb']=.0001
        if 'using_load_ai_lazy' not in jy_opt:
            self.jy_opt['using_load_ai_lazy']=True
            self.jy_opt['using_load_ai_lazy_num_pickups']=(len(all_nodes)-2)/3
            self.jy_opt['using_load_ai_lazy_max_pickups']=3
            #print('self.jy_opt[using_load_ai_num_pickups]')
            #print(self.jy_opt['using_load_ai_num_pickups'])
            #input('---')
        self.actions_of_node=actions_of_node
        if self.actions_of_node==None:
            self.compute_actions_of_each_node()
        self.dual_vec=dual_vec
        self.max_actions_in_route=max_actions_in_route
        
        
        self.create_action_2_red_cost_dict()
        self.init_res_state=init_res_state
        my_init_label=self.create_init_label()
        self.my_efficient_frontier=efficient_frontier(self.all_nodes)
        self.my_efficient_frontier.alter_fronteir_given_new_element(my_init_label)
        self.unexpand_labels=SortedObjectList()

        
        
        self.add_label(my_init_label)
        self.call_expansion_algorihtm_till()
    
    def return_solution(self):
        my_states_ordered=self.my_final_label.my_states_ordered
        red_cost=self.my_final_label.red_cost
        list_of_nodes_in_shortest_path=[]
        list_of_actions_used_in_col=self.my_final_label.my_actions_ordered
        print('my_states_ordered')
        print(my_states_ordered)
        for s in my_states_ordered:
            list_of_nodes_in_shortest_path.append(s.node)
        print('list_of_nodes_in_shortest_path')
        print(list_of_nodes_in_shortest_path)
        print('red_cost')
        print(red_cost)
        #input('---')
        return [list_of_nodes_in_shortest_path, list_of_actions_used_in_col, red_cost]
    def call_expansion_algorihtm_till(self):

        my_final_label=None
        best_red_cost=1
        #print('len(self.nodes)')
        #print(len(self.all_nodes))
        #input('---')
        while len(self.unexpand_labels)>0:
            best_label=self.unexpand_labels.pop()
            #print('expanding')
            #print('best_label.lb')
            #print(best_label.lb)
            #print('best_label.red_cost')
            #print(best_label.red_cost)
            #print('self.my_efficient_frontier.is_in_fronteir(best_label)')
            #print(self.my_efficient_frontier.is_in_fronteir(best_label))
            #print('self.all_nodes_ordered')
            #print(best_label.all_nodes_ordered)
            #input('---')
            if self.my_efficient_frontier.is_in_fronteir(best_label) ==False:
                continue
            if best_label.lb>=best_red_cost:
                print('breaking ')
                break
            all_new_labels=best_label.expand_label_fully()
            #print('len(all_new_labels)')
            #print(len(all_new_labels))
            for new_lab in all_new_labels:
                self.my_efficient_frontier.alter_fronteir_given_new_element(new_lab)
                if self.my_efficient_frontier.is_in_fronteir(new_lab):
                    #print('adding label')
                    self.add_label(new_lab)
                if new_lab.is_complete_route==True and  best_red_cost>new_lab.red_cost:
                    my_final_label=new_lab
                    best_red_cost=new_lab.red_cost
            if self.jy_opt['min_red_cost_terminate_early']>best_red_cost: 
                break
        #if my_final_label==None:
        #    input('no column found')
        print('best_red_cost')
        print(best_red_cost)
        print('len(self.unexpand_labels)')
        print(len(self.unexpand_labels))
        #input('---')
        self.my_final_label=my_final_label
        self.best_red_cost=best_red_cost
