import random
from typing import List
from src.common.action import Action
from src.common.state import State
from src.algorithm.update_states.state_update_function import StateUpdateFunction
from numpy import zeros, ones
from src.common.helper import Helper
from scipy.sparse import csr_matrix
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
        new_states:List[State] = self._generate_state_based_on_beta(this_beta,l_id) # generate possible states with given beta
        states_in_path:List[State] = self.get_states_from_action_list(list_of_actions,l_id) # generate states in path
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

        return beta

    def _generate_state_based_on_beta(self, beta: list,l_id):
        """This needs to be reviewed but basically this is the function that should be called in `get_new_states`."""
        myState = []
        dem_list = []
        #null_action = []
        node_to_states = {node:[State] for node in self.nodes}
        self.states_used_in_this_col=[]
        for idx in range(len(beta)):
            customer = beta[idx]
            myCanVisit = {
                f'can_visit: {u}': 0 if beta.index(u) < beta.index(customer) else 1
                for u in self.nodes if u not in {-1, -2}
            }
            if idx>0:
                dem_list.append(self.demands[beta[idx-1]])
            minimum_dem_remain = self.capacity - self.demands[customer]
            poss_demand_used = self.get_unique_value_from_list(dem_list,minimum_dem_remain)
            poss_demand_used.append(0)
            for d in poss_demand_used:
                resource_vector = myCanVisit.copy()
                resource_vector['cap_remain'] = self.capacity-d
                _, res_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,resource_vector)

                this_state = State(customer,res_vec,l_id,False,False)
                myState.append(this_state)
                if (d==0):
                    self.states_used_in_this_col.append(this_state)
                node_to_states[customer].append(this_state)
        source_state_resource_vector = {
            f'can_visit: {u}': 1 for u in self.nodes if u not in {-1, -2}}
        source_state_resource_vector['cap_remain'] = self.capacity
        _,source_state_res_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,source_state_resource_vector)
        source_state = State(-1, source_state_res_vec, l_id,True,False)
        sink_state_resource_vector = {
            f'can_visit: {u}': 0 for u in self.nodes if u not in {-1, -2}}
        sink_state_resource_vector['cap_remain'] = 0
        _,sink_state_res_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,sink_state_resource_vector)
        sink_state = State(-2, sink_state_res_vec, l_id,False,True)
        myState.append(source_state)
        myState.append(sink_state)
        self.states_used_in_this_col.append(source_state)
        self.states_used_in_this_col.append(sink_state)

        """calculate null actions"""
        # default_trans_min_input = {}
        # default_trans_term_add = {}
        # default_trans_term_min = {}
        # for u in self.nodes:
        #     default_trans_min_input[f'can_visit: {u}'] = 0
        #     default_trans_term_add[f'can_visit: {u}'] = 1
        #     default_trans_term_min[f'can_visit: {u}'] = 0
        
        # for node in node_to_states:
        #     list_of_states = node_to_states[node]
        #     for state1 in list_of_states:
        #         for state2 in list_of_states:
        #             if state1 != state2 and state1.state_vec['cap_remain'] > state2.state_vec['cap_remain']:
        #                 contribution_vector = zeros(len(self.nodes))
        #                 contribution_vector[node]=1
        #                 trans_min_input = default_trans_min_input
        #                 trans_term_add = default_trans_term_add
        #                 trans_term_min = default_trans_term_min
        #                 node_tail = node
        #                 node_head = node
        #                 cost = 0
        #                 this_null_action = Action(trans_min_input,trans_term_add,trans_term_min,node_tail,node_head,contribution_vector,cost)
        #                 null_action.append(this_null_action)
        return myState

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
    
    def get_unique_value_from_list(self,cus_lst, m):
        possible_sums = {0}  # Start with an empty subset sum
        dem_list = []
        
        for num in cus_lst:
            new_sums = set()
            for current_sum in possible_sums:
                new_sum = current_sum + num
                if new_sum <= m:
                    new_sums.add(new_sum)
            possible_sums.update(new_sums)
        
        possible_sums.discard(0)  # Remove the initial empty sum if not needed
        return sorted(possible_sums)  # Return sorted results if needed
    