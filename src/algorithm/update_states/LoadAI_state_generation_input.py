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
        self.pickup_node = self.pickup_to_dropoff.keys()
        self.dropoff_node = self.dropoff_to_pickup.keys()
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
    def _generate_beta_time(self,list_of_customer,list_of_action,initial_state_resource):
        self.betaTime = defaultdict()
        time_start = initial_state_resource['time']
        for a in list_of_action:
            this_res_vec = a.resource_consumption_vec
            time_consume = this_res_vec[0,self.resource_name_to_index['time']]
            beta_time_value = min(time_start + time_consume, self.time_window_start[a.node_head])
            self.betaTime[a.node_head] = beta_time_value 
        
        for node in self.nodes:
            if node not in list_of_customer and node in self.pickup_node and node not in {-1,-2}:
                this_time = random.randint(self.time_window_end[node],self.time_window_start[node])
                self.betaTime[node] = this_time
        self.betaTime[-1] = np.inf
        self.betaTime[-2] = 0
        print('finish beta time')
    def _generate_node_min_term(self, pickup_nodes,drop_off_nodes):
        
        min_term_vec_dict = defaultdict()
        for this_pickup_nodes in self.nodes:
            if this_pickup_nodes in self.pickup_node:
                minterm_vec = np.ones(self.number_of_resources)
                minterm_vec[self.resource_name_to_index['weight']] = np.inf
                minterm_vec[self.resource_name_to_index['volume']] = np.inf
                minterm_vec[self.resource_name_to_index['time']] = np.inf
                minterm_vec[self.resource_name_to_index['max_combined_loads']] = np.inf
                for other_pickup_node in pickup_nodes:
                    if self.betaTime[this_pickup_nodes] > self.betaTime[other_pickup_node]:
                        minterm_vec[self.resource_name_to_index[str((f'may_pickup',other_pickup_node))]] = 0
                
                
                for drop_off_node in drop_off_nodes:
                    minterm_vec[self.resource_name_to_index[str((f'may_avoid_dropoff',drop_off_node))]] = np.inf
                min_term_vec_dict[this_pickup_nodes] =minterm_vec
            else:
                minterm_vec = np.full((1,self.number_of_resources), np.inf)
                min_term_vec_dict[this_pickup_nodes] =minterm_vec
    
        return min_term_vec_dict

    def _get_input(self, list_of_customer, list_of_action, l_id, init_state):
        max_depth = self.capacity
        list_of_customer = list_of_customer[1:-1]
        #beta_dict, beta = self._generate_beta_term(list_of_customer)
        initial_state = init_state
        self._generate_beta_time(list_of_customer,list_of_action,initial_state)
        self.node_min_vec_dict = self._generate_node_min_term(self.pickup_node,self.dropoff_node)
        state_in_path = self.get_states_from_action_list(initial_state,list_of_action,l_id)
        
        
        
        action_reasonable = self._generate_reasonalbe_actions()
 
        
        depth_used = defaultdict()
        for actions in self.actions.values():
            for action in actions:
                if action.node_head in self.dropoff_to_pickup.keys():
                    depth_used[action] = 0
                else:
                    depth_used[action] = 1
        #return min_vec_dict, max_depth, min_vec_dict, action_reasonable
        user_ignore_state_action=None
        beta_info = {}
        beta_info['BetaTime'] = self.betaTime
        return max_depth, depth_used, state_in_path, self.node_min_vec_dict, action_reasonable, user_ignore_state_action,beta_info
    
    def _generate_reasonalbe_actions(self):
        
        num_pickups_keep=20
        #map each node to its pickup actions
        node_2_pickup_actions=dict()
        for u in self.nodes:
            node_2_pickup_actions[u]=[]
        for ai in self.actions:
            aL=self.actions[ai]
            for a in aL:
                n_tail=a.node_tail
                n_head=a.node_head
                my_cost=a.cost
 
                if n_head in self.pickup_to_dropoff:
                    
                    #chatGPT please add
                    node_2_pickup_actions[n_tail].append(tuple([a,my_cost]))
        for u in self.nodes:
            if u>-0.5:
                tmp=sorted(node_2_pickup_actions[n_tail], key=lambda x: x[1])
                tmp=tmp[0:num_pickups_keep]
                node_2_pickup_actions[n_tail]=tmp
        #chat gpt please remove from node_2_pickup_actions[n] all actions that are not in the K lowest cost actions
        #how do i get action from a sep
        #ittera
        #itterate over all actions and store for each action
 
        #comptue for each node the K nearest pickuop nodes
 
        reasonable_action = set()
        for pickup_node,dropoff_node in self.pickup_to_dropoff.items():
            reasonable_action.update(self.actions[pickup_node,dropoff_node])
        
        for u in self.nodes:
            for n in node_2_pickup_actions:
                for a in node_2_pickup_actions[n]:
                    reasonable_action.update(a)
 
        #for drop_off_node in self.dropoff_to_pickup.items():
        #    for node in self.neighbors[drop_off_node]:
        #        if node in self.pickup_to_dropoff.keys():
        #            reasonable_action.update(self.actions[pickup_node,dropoff_node])
        for u in self.pickup_node:
            for v in self.pickup_node:
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
    def get_states_from_action_list(self, initial_state,action_list: List[Action],l_id):
        """
        Returns a list of states given an action_list.
        """
        State_in_col=set()
        cur_state= State(action_list[0].node_tail,self.initial_resource_vector,l_id,action_list[0].node_tail == -1,action_list[0].node_head == -2)
        for a in action_list:
            new_s=a.get_head_state(cur_state, l_id)
            State_in_col.add(new_s)
            cur_state=new_s
        s_remove=[]
        s_add=[]
        for s in State_in_col:
        
            my_node=s.node
        
            my_state_vec=self.elementwise_min_csr(self.node_min_vec_dict [my_node],s.state_vec)
            s2=State(s.node,my_state_vec, l_id, s.node==-1,s.node==-2)# make a new state
            if np.sum(np.abs(s2.state_vec-s.state_vec))>.001:
                s_remove.append(s)
            else:
                s_add.append(s2)

        for s in s_remove:
            State_in_col.remove(s)
        for s in s_add:
            State_in_col.add(s)
        return State_in_col
    def elementwise_min_csr(self, vec1, vec2) -> csr_matrix:
        """
        Compute the elementwise minimum of a NumPy array and a CSR matrix.
        
        Parameters:
        -----------
        vec1 : numpy.ndarray
            First input vector/matrix
        vec2 : scipy.sparse.csr_matrix
            Second input vector/matrix
            
        Returns:
        --------
        scipy.sparse.csr_matrix
            A CSR matrix containing the elementwise minimum
        """
        vec1 = vec1.reshape(1, -1)
        # Check if shapes are compatible
        if vec1.shape != vec2.shape:
            raise ValueError(f"Matrices have incompatible shapes: {vec1.shape} vs {vec2.shape}")
        
        # Create a copy of vec2 to modify
        result = vec2.copy()
        
        # Get the indices of non-zero elements in vec2
        rows, cols = vec2.nonzero()
        
        # For each non-zero element in vec2, take the minimum with the corresponding element in vec1
        for i, j in zip(rows, cols):
            result[i, j] = min(vec1[i, j], vec2[i, j])
        
        # Find elements in vec1 that are non-zero but zero in vec2
        # Convert vec2 to a dense array for boolean comparison
        vec2_dense = vec2.toarray()
        mask = (vec2_dense == 0) & (vec1 != 0)
        additional_rows, additional_cols = np.where(mask)
        
        # Create lists to hold the new data
        new_data = []
        new_rows = []
        new_cols = []
        
        # Add the values from vec1 where vec2 is zero
        for i, j in zip(additional_rows, additional_cols):
            new_rows.append(i)
            new_cols.append(j)
            new_data.append(vec1[i, j])
        
        # If we have new values to add
        if new_data:
            # Convert current result to COO format for easier modification
            result_coo = result.tocoo()
            
            # Combine existing and new data
            combined_data = np.concatenate([result_coo.data, new_data])
            combined_rows = np.concatenate([result_coo.row, new_rows])
            combined_cols = np.concatenate([result_coo.col, new_cols])
            
            # Create a new CSR matrix
            result = csr_matrix((combined_data, (combined_rows, combined_cols)), shape=vec1.shape)
        
        return result