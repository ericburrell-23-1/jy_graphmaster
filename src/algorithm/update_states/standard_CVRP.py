import random
from typing import List
from src.common.action import Action
from src.common.state import State
from src.algorithm.update_states.state_update_function import StateUpdateFunction
from numpy import zeros, ones
from src.common.helper import Helper
from scipy.sparse import csr_matrix
from collections import defaultdict
import numpy as np
import random

class CVRP_state_update_function(StateUpdateFunction):
    """This module is very important!!! It tells us how we will update `res_states` after pricing!!!
    
    This is definitely not complete. Just has some code to give an idea of how it will look when it is done. Needs to be fixed. Might need additional inputs.
    
    Keep in mind this module looks different for every problem type. This is just for CVRP!"""
    def __init__(self, nodes, actions, capacity, demands, neighbors_by_distance, initial_resource_vector,resource_name_to_index,number_of_resources):
        super().__init__(nodes,actions)
        self.capacity = capacity
        self.demands = demands
        self.neighbors_by_distance = neighbors_by_distance
        self.initial_resource_vector = initial_resource_vector
        self.resource_name_to_index = resource_name_to_index
        self.number_of_resources = number_of_resources
        

    def get_states_from_random_beta(self, customer_list,l_id):
        this_beta = customer_list[1:-1]
        random.seed(l_id*1000)
        random.shuffle(this_beta)
        new_states:List[State] = self._generate_state_based_on_beta(this_beta,l_id)
        return [-1]+this_beta+[-2],new_states,self.states_used_in_this_col

    def get_new_states(self, list_of_customer, list_of_actions,l_id):

        this_beta = self._generate_beta_term(list_of_customer[1:-1]) #attached customer not in path into nearest customer in path
        
        new_states:List[State] = self._generate_state_based_on_beta_2(this_beta,l_id) # generate possible states with given beta
        states_in_path:List[State] = self.get_states_from_action_list(list_of_actions,l_id) # generate states in path
        for state in states_in_path:
            if state not in new_states:
                state.pretty_print_state()
        return [-1]+this_beta+[-2], new_states,  states_in_path
    
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
        
        beta_dict = defaultdict()
        for idx, customer in enumerate(beta):
            beta_dict[customer] = idx
        beta_dict[-1] = -np.inf
        beta_dict[-2] = np.inf
        return beta_dict

    def _generate_state_based_on_beta_2(self, beta:list,l_id):
        this_beta = beta
        full_resource_dict = np.ones(self.number_of_resources)
        full_resource_dict[0] = self.capacity
        full_resource_vec = csr_matrix(full_resource_dict.reshape(1, -1))
        empty_resource_dict = np.zeros(self.number_of_resources)
        empty_resource_vec = csr_matrix(empty_resource_dict.reshape(1, -1))
 
        #one for the source
        source_state = State(-1,full_resource_vec,l_id,True,False)
 
        #one for the sink
        sink_state = State(-2,empty_resource_vec,l_id,False,True)
        all_state = []
        all_state.append(source_state)
        all_state.append(sink_state)

        my_state = {source_state, sink_state}
        for u in self.nodes:
            dem_Threh = set()
            for w in self.nodes:
                if this_beta[w] < this_beta[u]:
                    for s in my_state:
                        if s.node == w and s.state_vec[0,0] > self.demands[u]+self.demands[w]:
                            new_cap_rem = s.state_vec[0,0] - self.demands[w]
                            dem_Threh.add(new_cap_rem)
            can_visit = defaultdict(list)
            for v in self.nodes:
                can_visit[v] = [0 if this_beta[u]<this_beta[v] else 1
                    for u in self.nodes if u not in {-1, -2}]
            for dem in dem_Threh:
                for v in self.nodes:
                    if v not in (-1,-2):
                        state_vec = [dem]
                        state_vec.extend(can_visit[v])
                        state_vec = csr_matrix(np.array(state_vec).reshape(1, -1))
                        this_state = State(u,state_vec,l_id,None,None)
                        all_state.append(this_state)


        return all_state
    def _generate_state_based_on_beta(self, beta: list, l_id):
        """
        Generate all possible states for a given sequence beta.
        A state is defined as (node, demand_remain) where demand_remain represents 
        the accumulated demand of the visited nodes.
        """
        all_states = []
        node_to_states = {node: [] for node in self.nodes}
        self.states_used_in_this_col = []
        
        # Process each node in the sequence
        for idx, customer in enumerate(beta):
            # First, let's consider all possible paths that *end* at this customer
            # This means generating states for all valid subpaths ending at customer
            
            # For all possible subpaths, we need to track which nodes would have been visited
            # By the time we get to this node, any node that comes before it in beta
            # and is included in our subpath would be marked as visited
            
            # Get all possible previous customers that might have been visited
            # These are nodes that appear before the current customer in beta
            previous_customers = beta[:idx]
            
            # For each previous node, we can either visit it or not
            # We need to generate states for all these combinations
            # Let's start with a simple approach by generating all possible subpaths
            from itertools import combinations, chain
            
            # Generate all possible combinations of previous nodes
            # We'll use these to create states for all possible subpaths
            all_prev_combs = []
            for i in range(len(previous_customers) + 1):
                all_prev_combs.extend(combinations(previous_customers, i))
                
            # For each combination of previous nodes, generate a state
            for prev_comb in all_prev_combs:
                # Create the can_visit vector
                # Nodes that are in prev_comb have been visited (can_visit = 0)
                # The current node (customer) is still visitable (can_visit = 1)
                # All other nodes are visitable (can_visit = 1)
                myCanVisit = {
                    f'can_visit: {u}': 0 if u in prev_comb else 1
                    for u in self.nodes if u not in {-1, -2}
                }
                
                # Calculate the accumulated demand from this subpath
                prev_demand = sum(self.demands[node] for node in prev_comb)
                
                # If this demand exceeds capacity, skip this subpath
                if prev_demand > self.capacity:
                    continue
                    
                # The remaining capacity after visiting all nodes in prev_comb
                remaining_cap = self.capacity - prev_demand
                
                # Create the resource vector
                resource_vector = myCanVisit.copy()
                resource_vector['cap_remain'] = remaining_cap
                
                # Convert to vector format
                _, res_vec = Helper.dict_2_vec(
                    self.resource_name_to_index, 
                    self.number_of_resources, 
                    resource_vector
                )
                
                # Create the state
                this_state = State(customer, res_vec, l_id, False, False)
                all_states.append(this_state)
                
                # Track states that start paths (with no previous demand)
                if len(prev_comb) == 0:
                    self.states_used_in_this_col.append(this_state)
                    
                # Add to node-to-states mapping
                node_to_states[customer].append(this_state)
        
        # Create source state (depot)
        source_state_resource_vector = {
            f'can_visit: {u}': 1 for u in self.nodes if u not in {-1, -2}
        }
        source_state_resource_vector['cap_remain'] = self.capacity
        _, source_state_res_vec = Helper.dict_2_vec(
            self.resource_name_to_index,
            self.number_of_resources,
            source_state_resource_vector
        )
        source_state = State(-1, source_state_res_vec, l_id, True, False)
        
        # Create sink state (return to depot)
        sink_state_resource_vector = {
            f'can_visit: {u}': 0 for u in self.nodes if u not in {-1, -2}
        }
        sink_state_resource_vector['cap_remain'] = 0
        _, sink_state_res_vec = Helper.dict_2_vec(
            self.resource_name_to_index,
            self.number_of_resources,
            sink_state_resource_vector
        )
        sink_state = State(-2, sink_state_res_vec, l_id, False, True)
        
        # Add source and sink states
        all_states.append(source_state)
        all_states.append(sink_state)
        self.states_used_in_this_col.append(source_state)
        self.states_used_in_this_col.append(sink_state)
        
        return all_states

    def get_states_from_action_list(self, action_list: List[Action],l_id):
        """
        Returns a list of states given an action_list.
        """
        if not action_list:
            return []
        states_list = []
        current_resources = self.initial_resource_vector.copy()
        #_,current_resources = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,current_resources)
        pred_state = State(action_list[0].node_tail,current_resources,l_id,action_list[0].node_tail == -1,action_list[0].node_head == -2)
        states_list.append(pred_state)
        for action in action_list:
            
            new_state = action.get_head_state(pred_state,l_id)
            
            # if new_resource_vector is None:
            #     print(f"Invalid resource transition from {action.node_head} to {action.node_tail}")
            #     break
            # new_state = State(action.node_head,current_resources,l_id,action.node_head == -1,action.node_head == -2)
            
            states_list.append(new_state)
            pred_state = new_state
            
        return states_list
        
    def get_unique_value_from_list(self, cus_lst, m):
        """
        Generate all possible unique sums from the list that don't exceed m.
        Uses dynamic programming approach for subset sum problem.
        
        Args:
            cus_lst: List of customer demands
            m: Maximum capacity
            
        Returns:
            List of all possible unique sums
        """
        possible_sums = {0}  # Start with an empty subset sum
        
        for num in cus_lst:
            new_sums = set()
            for current_sum in possible_sums:
                new_sum = current_sum + num
                if new_sum <= m:
                    new_sums.add(new_sum)
            possible_sums.update(new_sums)
        
        possible_sums.discard(0)  # Remove the initial empty sum if not needed
        return sorted(possible_sums)  # Return sorted results
    