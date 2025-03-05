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

        beta_dict, beta_list = self._generate_beta_term(list_of_customer[1:-1]) #attached customer not in path into nearest customer in path
        
        #new_states:List[State] = self._generate_state_based_on_beta_2_and_states_path(beta_list,beta_dict,l_id) # generate possible states with given beta
        states_in_path:List[State] = self.get_states_from_action_list(list_of_actions,l_id) # generate states in path
        beta_list,states_for_new_graph,states_in_path_to_return = self._generate_state_based_on_beta_2_and_ndoes_path(beta_list,beta_dict,l_id,states_in_path,list_of_actions)
        for state in states_in_path_to_return:
            if state not in states_for_new_graph:
                state.pretty_print_state()

        return beta_list, states_for_new_graph,  states_in_path_to_return
    
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

    def _generate_state_based_on_beta_2_and_ndoes_path(self, beta_list:list,beta_dict,l_id,states_from_path,ordered_actions):
        #print('beta_dict')
        #print(beta_dict)
        #print('beta list')
        #print(beta_list)
        #
        #print('statees made ')
        for si in range(0,len(states_from_path)):
            s=states_from_path[si]
            u=s.node
            if u>-0.5:
                print('self.number_of_resources')
                print(self.number_of_resources)
                for w in range(1,self.number_of_resources):
                    #print('w')
                    #print(w)
                    #print('u')
                    ##print('beta_dict[w]')
                    #print(beta_dict[w])
                    #print('beta_dict[u]')
                    #print(beta_dict[u])
                    #print('s.state_vec')
                    ##print(s.state_vec)
                    
                    if beta_dict[w]>=beta_dict[u]:
                        if s.state_vec[0,w]==0:
                            input('error here in the initial route being inconsistent')
                        s.state_vec[0,w]=1
                    else:
                        s.state_vec[0,w]=0
            if u==-2:
                s.state_vec=s.state_vec*0
            s.pretty_print_state()
            print('beta_list')
            print(beta_list)
            #input('--')
        #input('test there')
       #nodes that describe the path
        source_state=states_from_path[0]
        sink_state=states_from_path[-1]
        states_for_new_graph=set()
        states_for_new_graph.add(source_state)
        states_for_new_graph.add(sink_state)

        #my_state = {source_state, sink_state}
        num_cust=len(beta_list)-2
        dem_list=defaultdict(set)
        dem_list[-1].add(self.capacity)

        #check
        #debug cehcks 
        if beta_list[0]!=-1:
            input('error here source wrong position')
        if beta_list[-1]!=-2: 
            input('error here sink wrong position')
        for i in range(0,len(beta_list)-1):
            u=beta_list[i]
            v=beta_list[i+1]
            if beta_dict[u]>=beta_dict[v]:
                input('error ehre')
        
        for i in range(1,num_cust+1):
            u=beta_list[i]
            dem_list[u]=set([])
            for j in range(0,i):
                w=beta_list[j]
                for d_w in dem_list[w]:
                    if d_w>=self.demands[w]+self.demands[u]:
                        dem_list[u].add(d_w-self.demands[w])

            #print('dem_list[u]')
            #print(dem_list[u])
            #print('u')
            #print(u)
            #print('i')
            #print(i)
            #input('----')
            #my_res_vec_base=dict()
            #my_res_vec = np.array([1 if beta_dict[beta_list[j]]>=beta_dict[beta_list[i]] else 0 for j in range(1,num_cust+1)])
            my_res_vec=np.zeros((1,self.number_of_resources))
            # for ik in range(1,self.number_of_resources):
            #     customer = beta_list[ik]
            #     if beta_dict[customer]>=beta_dict[u]:
            #         my_res_vec[0,beta_dict[customer]]=1
            # print('my_res_vec')
            # print(my_res_vec)
            # print('beta_list')
            # print(beta_list)
            for cust_num in range(1,self.number_of_resources):
                if beta_dict[u]<=beta_dict[cust_num]:
                    my_res_vec[0,cust_num]=1
            #input('-- HOLD HERE--')
            #base_rez_vec = csr_matrix(my_res_vec.reshape(1, -1))
            # for j in range(1,num_cust+1):
            #     u=beta_list[i]
            #     v=beta_list[j]
            
            #     my_res_vec[j]=int(beta_dict[v]>=beta_dict[u])
            # base_rez_vec = csr_matrix(my_res_vec_base.reshape(1, -1))

            for d in dem_list[u]:
                this_vec = my_res_vec.copy()
                #this_vec = np.insert(this_vec, 0, d)
                # my_res_vec=base_rez_vec.copy()
                this_vec[0,0]=d
                print('u')
                print(u)
                print('d')
                print(d)
                print('my_res_vec')
                print(my_res_vec)
                print('this_vec')
                print(this_vec)
                print('beta_list')
                print(beta_list)
                #input('--look here-')
                vec_added =  csr_matrix(this_vec.reshape(1, -1))
                my_node=u
                is_source=False
                is_sink=False
                my_state=State(my_node, vec_added, l_id,is_source,is_sink)
                states_for_new_graph.add(my_state)

                if u==3 and d==8:
                    s=states_from_path[2]
                    s.pretty_print_state()
                    my_state.pretty_print_state()
                    tst=s.equals_minus_id(my_state)
                    #print('tst')
                    #print(tst)
                    #input('hold look curucial')
            #if u==3:    
            #    print('states above')
            #    input('new state')

        states_in_path_to_return=[]
        for s in states_from_path:
            state_out=self.helper_get_state_slow(s,states_for_new_graph)
            states_in_path_to_return.append(state_out)
            if state_out not in states_for_new_graph:
                input('error here')
        #debug
        for i in range(0,len(ordered_actions)):
            s1=states_in_path_to_return[i]
            s2=states_in_path_to_return[i+1]
            my_act=ordered_actions[i]
            my_act.check_valid(s1,s2)

        #print('ok im doing clearance here')
        for s in states_in_path_to_return:
            if s not in states_for_new_graph:
                input('error check here')
            else:
                print(' I found ')
                s.pretty_print_state()
        #print('DONE  im doing clearance here')

        return [beta_list,states_for_new_graph,states_in_path_to_return]




    def _generate_state_based_on_beta_2_and_states_path(self, beta_list:list,beta_dict,l_id,states_from_path,ordered_actions):
        
       #full_resource_dict = np.ones(self.number_of_resources)
       # full_resource_dict[0] = self.capacity
       # full_resource_vec = csr_matrix(full_resource_dict.reshape(1, -1))
       # empty_resource_dict = np.zeros(self.number_of_resources)
       # empty_resource_vec = csr_matrix(empty_resource_dict.reshape(1, -1))
        source_state=states_from_path[0]
        sink_state=states_from_path[-1]
        states_for_new_graph=[]
        states_for_new_graph.append(source_state)
        states_for_new_graph.append(sink_state)

        #my_state = {source_state, sink_state}
        num_cust=len(beta_list)-2
        dem_list=defaultdict(set)
        dem_list[-1].add(self.capacity)

        #check
        #debug cehcks 
        if beta_list[0]!=-1:
            input('error here source wrong position')
        if beta_list[-1]!=-2: 
            input('error here sink wrong position')
        for i in range(0,len(beta_list)-1):
            u=beta_list[i]
            v=beta_list[i+1]
            if beta_dict[u]>=beta_dict[v]:
                input('error ehre')
        
        for i in range(1,num_cust+1):
            u=beta_list[i]
            dem_list[u]=set([])
            for j in range(0,i):
                w=beta_list[j]
                for d_w in dem_list[w]:
                    if d_w>=self.demands[w]+self.demands[u]:
                        dem_list[u].add(d_w-self.demands[w])
            #my_res_vec_base=dict()
            my_res_vec = np.array([1 if beta_dict[beta_list[j]]>=beta_dict[beta_list[i]] else 0 for j in range(1,num_cust+1)])
            
            #base_rez_vec = csr_matrix(my_res_vec.reshape(1, -1))
            # for j in range(1,num_cust+1):
            #     u=beta_list[i]
            #     v=beta_list[j]
            
            #     my_res_vec[j]=int(beta_dict[v]>=beta_dict[u])
            # base_rez_vec = csr_matrix(my_res_vec_base.reshape(1, -1))

            for d in dem_list[u]:
                this_vec = my_res_vec[:]
                this_vec = np.insert(this_vec, 0, d)
                # my_res_vec=base_rez_vec.copy()
                # my_res_vec[0]=d
                vec_added =  csr_matrix(this_vec.reshape(1, -1))
                my_node=u
                is_source=False
                is_sink=False
                my_state=State(my_node, vec_added, l_id,is_source,is_sink)
                states_for_new_graph.append(my_state)

        states_in_path_to_return=[]
        for s in states_from_path:
            state_out=self.helper_get_state_slow(s,states_for_new_graph)
            states_in_path_to_return.append(state_out)

        #debug
        for i in range(0,len(ordered_actions)):
            s1=states_in_path_to_return[i]
            s2=states_in_path_to_return[i+1]
            my_act=ordered_actions[i]
            my_act.check_valid(s1,s2)


        return [beta_list,states_for_new_graph,states_in_path_to_return]

    def helper_get_state_slow(self,desired_state,states_from_beta):
        state_2_return=None
        if desired_state.is_source==True:
            for s in states_from_beta:
                if s.is_source==True:
                    state_2_return=s
                    break
            if  state_2_return==None:
                input('error here no source foudn')
        if desired_state.is_sink==True:
            for s in states_from_beta:
                if s.is_sink==True:
                    state_2_return=s
                    break
            if  state_2_return==None:
                input('error here no sink found')
        if state_2_return==None:
            for s in states_from_beta:
                if desired_state.equals_minus_id(s):
                    state_2_return=s
                    break
            if  state_2_return==None:
                desired_state.pretty_print_state()

                print('----')
                input('error here no state found')

        return state_2_return

        
    
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
    