import random
from typing import List
from src.common.action import Action
from src.common.state import State
from numpy import zeros, ones
from src.common.helper import Helper
from scipy.sparse import csr_matrix
from collections import defaultdict
import numpy as np

class CVRP_state_input():
    """This module is very important!!! It tells us how we will update `res_states` after pricing!!!
    
    This is definitely not complete. Just has some code to give an idea of how it will look when it is done. Needs to be fixed. Might need additional inputs.
    
    Keep in mind this module looks different for every problem type. This is just for CVRP!"""

    def __init__(self, nodes, actions, capacity, demands, neighbors_by_distance, neighbors, initial_resource_vector,resource_name_to_index,number_of_resources):
        self.nodes = nodes
        self.actions = actions
        self.capacity = capacity
        self.demands = demands
        self.neighbors_by_distance = neighbors_by_distance
        self.neighbors = neighbors
        self.initial_resource_vector = initial_resource_vector
        self.resource_name_to_index = resource_name_to_index
        self.number_of_resources = number_of_resources
        random.seed(1000)

    def get_states_from_random_beta(self, customer_list,l_id):
        this_beta = customer_list[1:-1]
        random.shuffle(this_beta)
        new_states:List[State] = self._generate_state_based_on_beta(this_beta,l_id)
        return [-1]+this_beta+[-2],new_states,self.states_used_in_this_col

    
    def _generate_beta_term(self, list_of_customer):
        idx_of_customer = {u: -1 for u in self.nodes if u not in {-1,-2}}
        for idx in range(len(list_of_customer)):
            idx_of_customer[list_of_customer[idx]] = idx
        customer_not_in_route = list(set(self.nodes)-set(list_of_customer)-{-1,-2})
        for customer in customer_not_in_route:
            for customer2 in self.neighbors_by_distance[customer]:
                if customer2 in list_of_customer:
                    idx_of_customer[customer] = idx_of_customer[customer2] + \
                        round(random.random(), 5)*0.01
                    break
        beta = sorted(idx_of_customer, key=lambda k: idx_of_customer[k])
        beta = [-1] + beta + [-2]
        beta_dict = defaultdict()
        for idx, customer in enumerate(beta):
            beta_dict[customer] = idx
        beta_dict[-1] = -np.inf
        beta_dict[-2] = np.inf
        return beta_dict, beta

    def _generate_node_min_term(self, beta,beta_dict):
        min_term_vec_dict = defaultdict()
        
        for node in self.nodes:
            min_term_vec = np.ones(self.number_of_resources)
            min_term_vec[0] = self.capacity
            for other_node in self.nodes:
                if other_node not in (-1,-2) and beta_dict[other_node] < beta_dict[node] and other_node != node:
                    min_term_vec[other_node] = 0
            min_term_vec_dict[node] = csr_matrix(min_term_vec)
    
        return min_term_vec_dict

    def _get_input(self, list_of_customer, list_of_action, l_id):
        max_depth = self.capacity
        action_reasonable = set()
        for n1 in self.nodes:
            for n2 in self.neighbors[n1]:
                if (n1,n2) in self.actions.keys():
                    try:
                        action_reasonable.update(set(self.actions[(n1,n2)]))
                    except:
                        input('check here')
        list_of_customer = list_of_customer[1:-1]
        beta_dict, beta = self._generate_beta_term(list_of_customer)
        min_vec_dict = self._generate_node_min_term(beta,beta_dict)
        state_in_path = self.get_states_from_action_list(list_of_action,l_id,beta_dict)
        depth_used = defaultdict()
        for actions in self.actions.values():
            for action in actions:
                depth_used[action] = 1
        #return min_vec_dict, max_depth, min_vec_dict, action_reasonable
        user_ignore_state_action=None
        return max_depth, depth_used, state_in_path, min_vec_dict, action_reasonable, user_ignore_state_action, beta, beta_dict
    
    def get_states_from_action_list(self, action_list: List[Action],l_id, beta_dict):
        """
        Returns a list of states given an action_list.
        """
        if not action_list:
            return []
        states_list = []
        current_resources = self.initial_resource_vector.copy()
        #_,current_resources = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,current_resources)
        source_vec = np.ones(self.number_of_resources)
        source_vec[0]=self.capacity
        source_vec = csr_matrix(source_vec)
        pred_state = State(action_list[0].node_tail,source_vec,l_id,action_list[0].node_tail == -1,action_list[0].node_head == -2)
        states_list.append(pred_state)
        for action in action_list:
            
            new_state = action.get_head_state(pred_state,l_id)
            _, idx_list = new_state.state_vec.nonzero()
            for idx in idx_list:
                if idx >0 and beta_dict[idx] < beta_dict[new_state.node]:
                    new_state.state_vec[0,idx] = 0

            if new_state == None:
                
                #input('check here, none state generated from path')
                continue
            # if new_resource_vector is None:
            #     print(f"Invalid resource transition from {action.node_head} to {action.node_tail}")
            #     break
            # new_state = State(action.node_head,current_resources,l_id,action.node_head == -1,action.node_head == -2)
            
            states_list.append(new_state)
            pred_state = new_state
        sink_vec  = np.zeros(self.number_of_resources)
        sink_vec = csr_matrix(sink_vec)
        pred_state = State(action_list[-1].node_head,sink_vec,l_id,action_list[-1].node_tail == -1,action_list[0].node_head == -2)
        return states_list