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
    
 
    def __init__(self, nodes, actions, capacity, demands, time_window_start, time_window_end, pickup_to_dropoff, dropoff_to_pickup, neighbors_by_distance, neighbors, travel_time, initial_resource_vector, resource_name_to_index, number_of_resources,problem_info):
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
        self.problem_info = problem_info
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
            time_start = beta_time_value
        
        for node in self.nodes:
            if node not in list_of_customer and node not in {-1,-2}:
                if node in self.time_window_end:
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
                    #TODO: double check here if it is less or greater
                    if self.betaTime[this_pickup_nodes] < self.betaTime[other_pickup_node]:
                        minterm_vec[self.resource_name_to_index[str((f'may_pickup',other_pickup_node))]] = 0
                
                
                for drop_off_node in drop_off_nodes:
                    minterm_vec[self.resource_name_to_index[str((f'may_avoid_dropoff',drop_off_node))]] = np.inf
                min_term_vec_dict[this_pickup_nodes] =csr_matrix(minterm_vec)
            elif this_pickup_nodes in self.dropoff_node:
                minterm_vec = np.full((1,self.number_of_resources), np.inf)
                for other_pickup_node in pickup_nodes:

                    try:
                        if self.betaTime[this_pickup_nodes] < self.betaTime[other_pickup_node]:
                            minterm_vec[0,self.resource_name_to_index[str((f'may_pickup',other_pickup_node))]] = 0
                    except:
                        print('check here')
                    min_term_vec_dict[this_pickup_nodes] =csr_matrix(minterm_vec)
            else:
                minterm_vec = np.full((1,self.number_of_resources), np.inf)
                min_term_vec_dict[this_pickup_nodes] =csr_matrix(minterm_vec)
    
        return min_term_vec_dict

    def _get_input(self, list_of_customer, list_of_action, l_id, init_state_resource):
        max_depth = self.capacity
        list_of_customer = list_of_customer[1:-1]
        #beta_dict, beta = self._generate_beta_term(list_of_customer)
        initial_state = init_state_resource
        self._generate_beta_time(list_of_customer,list_of_action,initial_state)
        self.node_min_vec_dict = self._generate_node_min_term(self.pickup_node,self.dropoff_node)
        state_in_path = self.get_states_from_action_list(list_of_customer,list_of_action,l_id)
        
        for idx in range(len(state_in_path)-1):
            s1 = state_in_path[idx]
            s2 = state_in_path[idx+1]
            a = list_of_action[idx]
            if a.check_valid(s1,s2)==False:
                input('error here')

        self.action_reasonable, self.action_reasonable_dict = self._generate_reasonalbe_actions()
 
        
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
        return max_depth, depth_used, state_in_path, self.node_min_vec_dict, self.action_reasonable,self.action_reasonable_dict, user_ignore_state_action,beta_info

    def _generate_reasonalbe_actions(self):
        
        reasonable_action_dict = defaultdict()
        reasonable_action = set()
        #every action evolve source or sink
        for u in self.pickup_node:
            reasonable_action.update(self.actions[(-1,u)])
            reasonable_action_dict[(-1,u)] = self.actions[(-1,u)]
        for v in self.dropoff_node:
            reasonable_action.update(self.actions[(v,-2)])
            reasonable_action_dict[(v,-2)] = self.actions[(v,-2)]
        # action from pickup to dropoff
        for pickup_node in self.pickup_node:

            for dropoff_node in self.dropoff_node:
                if (pickup_node,dropoff_node) in self.actions.keys():

                    this_action = self.actions[(pickup_node,dropoff_node)]
                    reasonable_action.update(this_action)
                    reasonable_action_dict[(pickup_node,dropoff_node)] = this_action
        # action from dropoff to dropoff
        for dropoff_node_1 in self.dropoff_node:
            for dropoff_node_2 in self.dropoff_node:
                if (dropoff_node_1,dropoff_node_2) in self.actions.keys() and dropoff_node_1 != dropoff_node_2:

                    this_action = self.actions[(dropoff_node_1,dropoff_node_2)]
                    reasonable_action.update(this_action)
                    reasonable_action_dict[(dropoff_node_1,dropoff_node_2)] = this_action

        # action from dropoff to nearby pickup if it is neighbor
        for u in self.dropoff_node:
            for v in set(self.neighbors[u]) & self.pickup_node:
                if (u,v) in self.actions.keys():
                    reasonable_action.update(self.actions[(u,v)])
                    reasonable_action_dict[(u,v)] = self.actions[(u,v)]
        #for drop_off_node in self.dropoff_to_pickup.items():
        #    for node in self.neighbors[drop_off_node]:
        #        if node in self.pickup_to_dropoff.keys():
        #            reasonable_action.update(self.actions[pickup_node,dropoff_node])

        for u in self.pickup_node:
            for v in self.pickup_node:
                if u!=v:
                    """calculate cost_uvuv"""
                    cost_uvuv = self.actions[(u,v)][0].cost + \
                        self.actions[(v,self.pickup_to_dropoff[u])][0].cost+ self.actions[(self.pickup_to_dropoff[u],self.pickup_to_dropoff[v])][0].cost
                    route = [-1,u,v,self.pickup_to_dropoff[u],self.pickup_to_dropoff[v],-2]
                    s = State(-1,self.initial_resource_vector,0,True,False)
                    for n1,n2 in zip(route[:-1],route[1:]):
                        a = self.actions[(n1,n2)][0]
                        new_s = a.get_head_state(s,1)
                        if new_s == None:
                            cost_uvuv = np.inf
                            break
                        s = new_s
                        
                    """calculate cost_uuvv"""

                    cost_uuvv = self.actions[(u,self.pickup_to_dropoff[u])][0].cost + \
                        self.actions[(self.pickup_to_dropoff[u],v)][0].cost + self.actions[(v,self.pickup_to_dropoff[v])][0].cost
                    route = [-1,u,self.pickup_to_dropoff[u],v,self.pickup_to_dropoff[v],-2]

                    s = State(-1,self.initial_resource_vector,0,True,False)
                    for n1,n2 in zip(route[:-1],route[1:]):
                        a = self.actions[(n1,n2)][0]
                        new_s = a.get_head_state(s,1)
                        if new_s == None:
                            cost_uuvv = np.inf
                            break
                        s = new_s

                    min_of_two_cost = min(cost_uvuv,cost_uuvv)
                    cost_uu_vv = self.actions[(-1,u+2*len(self.dropoff_node))][0].cost + \
                        self.actions[(u+2*len(self.dropoff_node),-2)][0].cost + \
                            self.actions[(-1,v+2*len(self.dropoff_node))][0].cost+\
                                self.actions[(v+2*len(self.dropoff_node),-2)][0].cost
                    if min_of_two_cost < cost_uu_vv:
                        reasonable_action.update(self.actions[(u,v)])
                        reasonable_action_dict[(u,v)] = self.actions[(u,v)]
        #print('return reasonable action')
        for a in reasonable_action:
            if not isinstance(a, Action):
                print(a)
                input('error here')
        return reasonable_action, reasonable_action_dict
        
    def get_states_from_action_list(self, list_of_customer, action_list: List[Action], l_id):
        """
        Returns a list of states given an action_list, ensuring the states follow the 
        sequence in list_of_customer, with source (-1) at the beginning and sink (-2) at the end.
        """
        # Create initial state
        cur_state = State(action_list[0].node_tail, self.initial_resource_vector, l_id, 
                        action_list[0].node_tail == -1, action_list[0].node_head == -2)
        
        # Initialize ordered state list with the source state
        State_in_col = [cur_state]
        
        # Generate states in the original sequence
        for a in action_list:
            new_s = a.get_head_state(cur_state, l_id)
            if new_s == None:
                print('error here')
            State_in_col.append(new_s)
            cur_state = new_s
        
        return State_in_col
    def elementwise_min_csr(self, vec1, vec2) -> csr_matrix:
        """
        Compute the elementwise minimum of two CSR matrices.
        
        Parameters:
        -----------
        vec1 : scipy.sparse.csr_matrix
            First input sparse matrix
        vec2 : scipy.sparse.csr_matrix
            Second input sparse matrix
            
        Returns:
        --------
        scipy.sparse.csr_matrix
            A CSR matrix containing the elementwise minimum
        """
        # Check if shapes are compatible
        if vec1.shape != vec2.shape:
            raise ValueError(f"Matrices have incompatible shapes: {vec1.shape} vs {vec2.shape}")
        
        # Convert to COO format for easier manipulation
        cx1 = vec1.tocoo()
        cx2 = vec2.tocoo()
        
        # Create dictionaries for non-zero values
        dict1 = {(i, j): v for i, j, v in zip(cx1.row, cx1.col, cx1.data)}
        dict2 = {(i, j): v for i, j, v in zip(cx2.row, cx2.col, cx2.data)}
        
        # Combine keys
        all_keys = set(dict1.keys()).union(set(dict2.keys()))
        
        # Create new data for the minimum values
        rows, cols, data = [], [], []
        for i, j in all_keys:
            # Get values, with 0 as default for missing keys
            val1 = dict1.get((i, j), 0)
            val2 = dict2.get((i, j), 0)
            min_val = min(val1, val2)
            
            # Only include non-zero values in the result
            if min_val != 0:
                rows.append(i)
                cols.append(j)
                data.append(min_val)
        
        # Create a new CSR matrix
        return csr_matrix((data, (rows, cols)), shape=vec1.shape)

    def is_less_than_elementwise(self,matrix1, matrix2):
        """
        Checks if matrix1 is elementwise less than matrix2 for sparse matrices.
        Returns True if ALL elements in matrix1 are less than corresponding elements in matrix2.
        Returns False otherwise.
        """
        # Convert to arrays for elementwise comparison
        matrix1_array = matrix1.toarray()
        matrix2_array = matrix2.toarray()
        
        # Perform elementwise comparison
        comparison = matrix1_array < matrix2_array
        
        # Return True if ALL elements satisfy the condition
        return np.all(comparison)