import numpy as np
class Route:
    def __init__(self,state_action_alt_repeat,weight):

        self.Exog_vec_non_zero_indices = []
        self.Exog_vec_non_zero_val = []
        self.state_action_alt_repeat=state_action_alt_repeat #input is states and actions alternating
        self.clean_state_action_alt_repeat_by_removing_null()
        self.weight=weight #what is the corresponding amounf of this in the solution
        self.generate_states_nodes_actions_ordered()
        self.generate_all_node_pairs_ordered()
        self.generate_cost_exog_vector()
        #self.verify_feasibility()
        self.path_id = hash(tuple(self.node_in_ordered))
    def get_red_cost(self,dual):
        if len(self.Exog_vec_non_zero_indices) < 0.5:
            return self.cost
        
        return self.cost - np.dot(self.Exog_vec_non_zero_val, dual[self.Exog_vec_non_zero_indices])
    def clean_state_action_alt_repeat_by_removing_null(self):
        new_state_action_alt_repeat=[self.state_action_alt_repeat[0]]
        for act_ind in range(1,len(self.state_action_alt_repeat),2):
            my_action=self.state_action_alt_repeat[act_ind]
            if my_action.mark_of_null_action==False:
                new_state_action_alt_repeat.append(self.state_action_alt_repeat[act_ind])
                new_state_action_alt_repeat.append(self.state_action_alt_repeat[act_ind+1])
        self.state_action_alt_repeat=new_state_action_alt_repeat
    def generate_cost_exog_vector(self):
        
        
        self.cost=0
        #self.Exog_vec=self.just_actions_ordered[0].Exog_vec*0

        for i  in range(0,len(self.just_actions_ordered)):
            my_act=self.just_actions_ordered[i]
            self.cost=self.cost+my_act.cost
            #self.Exog_vec=self.Exog_vec+self.just_actions_ordered[i].Exog_vec
            non_zero_indices = my_act.red_cost_non_zero_cal_indices
            non_zero_val = my_act.red_cost_non_zero_cal_vals
            # if len(non_zero_indices)>0:
            if non_zero_indices!=None:
                if non_zero_indices not in self.Exog_vec_non_zero_indices:
                    self.Exog_vec_non_zero_indices.append(non_zero_indices)
                    self.Exog_vec_non_zero_val.append(non_zero_val)
                else:
                    index = self.Exog_vec_non_zero_indices.index(non_zero_indices)
                    self.Exog_vec_non_zero_val[index] += non_zero_val

            # for i,idx in enumerate(non_zero_indices):
            #     self.Exog_vec[idx] += self.Exog_vec[idx] + non_zero_val[i]

        
    def generate_states_nodes_actions_ordered(self):
        #generate all states and actions in order
        self.just_states_ordered=[]
        self.just_nodes_ordered=[]
        for i in range(0,len(self.state_action_alt_repeat),2):
            self.just_states_ordered.append(self.state_action_alt_repeat[i])
            self.just_nodes_ordered.append(self.state_action_alt_repeat[i].node)
        self.just_actions_ordered=[]
        
        for i in range(1,len(self.state_action_alt_repeat),2):
            self.just_actions_ordered.append(self.state_action_alt_repeat[i])
        #print('check here for state and action list')
        #print('len(self.just_states_ordered)')
        #print(len(self.just_states_ordered))
        #print('len(self.just_actions_ordered)')
        #print(len(self.just_actions_ordered))
        #input('---')
        #generate all node pairs
    def generate_subset_routes(self):
        route = self.node_in_ordered
        start_depot = route[0]
        end_depot = route[-1]
        
        # Extract pickup nodes from the route
        pickup_nodes = []
        
        for node in self.nodes_picked_up:  # Skip start/end depot
            pickup_nodes.append(node)
        
        # If fewer than 2 pickups, return empty list
        if len(pickup_nodes) < 2:
            return []
        
        # Generate all combinations of 2 pickup nodes
        valid_routes = []
        from itertools import combinations
        
        for pickup_combo in combinations(pickup_nodes, 2):
            # Calculate corresponding dropoffs
            dropoff_combo = tuple(p + len(self.pickup_node) for p in pickup_combo)
            
            # Create the new route - convert tuples to sets for union
            nodes_to_include = set(pickup_combo).union(set(dropoff_combo))
            
            # Add nodes in the original order
            new_route = [start_depot]
            for node in route[1:-1]:
                if node in nodes_to_include:
                    new_route.append(node)
            new_route.append(end_depot)
            
            valid_routes.append(new_route)
        
        return valid_routes
        
    def generate_all_node_pairs_ordered(self):
        self.all_node_pairs_ordered=set([])
        self.node_in_ordered = []
        self.nodes_picked_up =self.just_states_ordered[-1].picked_up
        for i in range(0,len(self.just_states_ordered)):
            s1=self.just_states_ordered[i]
            self.node_in_ordered.append(s1.node)

            self.all_node_pairs_ordered.add((s1.node,s1.node))
            for j in range(i+1,len(self.just_states_ordered)):
                s2=self.just_states_ordered[j]
                self.all_node_pairs_ordered.add((s1.node,s2.node))
    def verify_feasibility(self):

        #verify that hte route is feasible

        #check that first state is source
        flag=True
        if self.just_states_ordered[0].is_source==False:
            flag=False
            input('error here')
        if self.just_states_ordered[-1].is_sink==False:
            flag=False
            input('error here 2')
        #print('check here')
        for i in range(0,len(self.just_states_ordered)-1):
            s1=self.just_states_ordered[i]
            s2=self.just_states_ordered[i+1]
            my_act=self.just_actions_ordered[i]
            valid = my_act.check_valid(s1,s2)
            if valid == False:
                flag = False
                #input('action not valide for s1 s2 here')
        
        return flag
    def __eq__(self, other: 'Route') -> bool:
        if other is None:
            return False
        return self.path_id == other.path_id

    def __hash__(self) -> int:
       """
       Provides a hash so that State objects can be used in sets or as dictionary keys.
       We hash by the node and the contents of res_vec.
       """
       return self.path_id