import random
from typing import List
from src.common.action import Action
from src.common.state import State
from numpy import zeros, ones
from src.common.helper import Helper
from scipy.sparse import csr_matrix
from collections import defaultdict
import numpy as np

class LoadAI_state_input():
    

    def __init__(self, nodes, actions, capacity, demands, time_window_start, time_window_end, pickup_to_dropoff, dropoff_to_pickup, neighbors_by_distance, neighbors, travel_time, initial_resource_vector, resource_name_to_index, number_of_resources):
        self.nodes = nodes
        self.actions = actions
        self.capacity = capacity
        self.demands = demands
        self.time_window_start=time_window_start
        self.time_window_end=time_window_end
        self.pickup_to_dropoff=pickup_to_dropoff
        self.dropoff_to_pickup=dropoff_to_pickup
        self.neighbors_by_distance = neighbors_by_distance
        self.neighbors = neighbors
        self.travel_time = travel_time
        self.initial_resource_vector = initial_resource_vector
        self.resource_name_to_index = resource_name_to_index
        self.number_of_resources = number_of_resources
        random.seed(1000)


    
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

    def _generate_node_min_term(self, beta,beta_dict, pick_up_state,list_of_customer):
        self.betaTime = {state.node: state.time for state in pick_up_state}
        for node in self.nodes:
            if node not in list_of_customer and node in self.pickup_to_dropoff.keys():
                this_time = random.randint(self.time_window_start[node], self.time_window_drop[node])
                self.betaTime[node] = this_time
        min_term_vec_dict = defaultdict()
        
        
    
        return min_term_vec_dict

    def _get_input(self, list_of_customer, list_of_action, l_id):
        max_depth = self.capacity
        action_reasonable = self._generate_reasonalbe_actions()
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
    
    def _generate_reasonalbe_actions(self):
        reasonable_action = set()
        for pickup_node,dropoff_node in self.pickup_to_dropoff.items():
            reasonable_action.update(self.actions[pickup_node,dropoff_node])

        for drop_off_node in self.dropoff_to_pickup.items():
            for node in self.neighbors[drop_off_node]:
                if node in self.pickup_to_dropoff.keys():
                    reasonable_action.update(self.actions[pickup_node,dropoff_node])
        for u in self.dropoff_to_pickup.items():
            for v in self.dropoff_to_pickup.items():
                if u!=v:
                    """calculate cost_uvuv""" 
                    cost_uvuv = self.betaTime[u]
                    arrive_at_pickup_v = cost_uvuv+self.travel_time[(u,v)]
                    if arrive_at_pickup_v < self.time_window_end[v]:
                        cost_uvuv = max(arrive_at_pickup_v, self.time_window_start[v])
                    else:
                        cost_uvuv = np.inf
                        break
                    arrive_at_dropoff_u = cost_uvuv + self.travel_time[(v,self.pickup_to_dropoff[u])]
                    if arrive_at_dropoff_u < self.time_window_end[self.pickup_to_dropoff[u]]:
                        cost_uvuv = max(arrive_at_dropoff_u, self.time_window_start[self.pickup_to_dropoff[u]])
                    else:
                        cost_uvuv = np.inf
                        break
                    arrive_at_dropoff_v = cost_uvuv + self.travel_time[(self.pickup_to_dropoff[u],self.pickup_to_dropoff[v])]
                    if arrive_at_dropoff_v < self.time_window_end[self.pickup_to_dropoff[v]]:
                        cost_uvuv = max(arrive_at_dropoff_v, self.time_window_start[self.pickup_to_dropoff[v]])
                    else:
                        cost_uvuv = np.inf
                        break
                    """calculate cost_uuvv""" 

                    cost_uuvv = self.betaTime[u]
                    # Travel from pickup u to dropoff u
                    arrive_at_dropoff_u = cost_uuvv + self.travel_time[(u, self.pickup_to_dropoff[u])]
                    if arrive_at_dropoff_u < self.time_window_end[self.pickup_to_dropoff[u]]:
                        cost_uuvv = max(arrive_at_dropoff_u, self.time_window_start[self.pickup_to_dropoff[u]])
                    else:
                        cost_uuvv = np.inf

                    # Travel from dropoff u to pickup v
                    arrive_at_pickup_v = cost_uuvv + self.travel_time[(self.pickup_to_dropoff[u], v)]
                    if arrive_at_pickup_v < self.time_window_end[v]:
                        cost_uuvv = max(arrive_at_pickup_v, self.time_window_start[v])
                    else:
                        cost_uuvv = np.inf

                    # Travel from pickup v to dropoff v
                    arrive_at_dropoff_v = cost_uuvv + self.travel_time[(v, self.pickup_to_dropoff[v])]
                    if arrive_at_dropoff_v < self.time_window_end[self.pickup_to_dropoff[v]]:
                        cost_uuvv = max(arrive_at_dropoff_v, self.time_window_start[self.pickup_to_dropoff[v]])
                    else:
                        cost_uuvv = np.inf
                    if cost_uvuv < cost_uuvv:
                        reasonable_action.update(self.actions[u,v])
                        reasonable_action.update(self.actions[v,self.pickup_to_dropoff[u]])
                        reasonable_action.update(self.actions[self.pickup_to_dropoff[u],self.pickup_to_dropoff[v]])
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