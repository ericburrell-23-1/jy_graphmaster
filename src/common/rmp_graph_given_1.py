from collections import defaultdict

class RMP_graph_given_l:

    def __init__(self,my_Multi_Graph_Object,resStates_minus_by_node,res_actions):
        self.my_Multi_Graph_Object=my_Multi_Graph_Object#provides the id for hte l\in Omega_R generating this.  All nodes shoudl have the same id (even if we treate all nodes as being part of the same graph we can use this )
        self.resStates_minus_by_node=resStates_minus_by_node #dictionary that taks in the node and returns all states assocaited with that node in ResStates
        self.res_actions=res_actions #set that holds all actions that are possible.  Includes the Null action
        self.nodes=self.resStates_minus_by_node.keys()
    def initialize_system(self):
        
        for  u in self.my_Multi_Graph_Object.resStates_minus_by_node:
            self.update_domination(u)
            self.RMP_update_sub_compute_min_dominating_states_by_node(u)
            self.RMP_update_sub_compute_maximum_dominated_states_by_node(u)
        self.actions_s1_s2_non_dom = defaultdict(set)
        for a1 in self.res_actions:
           self.RMP_clean_states_EZ(self,a1)
        self.RMP_make_null_actions()
        self.RMP_compute_remove_redundant_actions()
        self.RMP_make_equiv_classes()
    def update_domination(self, my_node):
        """Updates domination relationships for all states of a given node."""

        # Ensure defaultdict(set) for automatic key initialization
        self.all_included_states_dominating_s = defaultdict(set)
        self.all_included_states_that_s_dominates = defaultdict(set)

        all_states = self.all_states_of_node[my_node]  # Avoid redundant dictionary lookup

        for s in all_states:
            self.all_included_states_dominating_s[s] = self.my_Multi_Graph_Object.state_2_is_dom_states_dict[s] & all_states
            self.all_included_states_that_s_dominates[s] = self.my_Multi_Graph_Object.state_2_dom_states_dict[s] & all_states


    def RMP_update_sub_compute_maximum_dominated_states_by_node(self,my_node):

        self.state_max_dom_dict=dict()#Crate place to store maximally dominated stats 
        
        for s in self.all_states_of_node[my_node]: #iterate over all states s
            do_remove=self.all_included_states_that_s_dominates|self.all_included_states_that_s_dominates[s]
            self.state_max_dom_dict[s]=self.all_included_states_that_s_dominates[s]-do_remove#create object to store states

    def RMP_update_sub_compute_min_dominating_states_by_node(self, my_node):
    
        self.state_min_dom_dict = {}  # Create dictionary to store minimally dominated states
        
        for s in self.all_states_of_node[my_node]:  # Iterate over all states in the node
            dominating_states = self.all_included_states_dominating_s[s]  # Avoid redundant lookups
            do_remove = self.all_included_states_dominating_s|dominating_states
            self.state_min_dom_dict[s] = dominating_states - do_remove  # Store minimally dominated states

    def RMP_compute_remove_redundant_actions(self):
        """Removes dominated actions from each (s1, s2) pair."""
        
        self.actions_s1_s2_clean = defaultdict(set)

        for s1, s2 in self.actions_s1_s2:  # Iterate over (s1, s2) pairs
            my_tup = (s1, s2)
            my_actions = self.actions_s1_s2_non_dom[my_tup]  # Get non-dominated actions
            do_remove = self.dom_actions_pairs| my_actions
            
            # Store cleaned actions (removing dominated ones)
            self.actions_s1_s2_clean[my_tup] = my_actions - do_remove

    def RMP_make_null_actions(self):  
        #makes null action terms.  This is for dropping resources 
        for s1 in self.state_max_dom_dict:
            for s2 in self.state_max_dom_dict[s1]:
                self.actions_s1_s2_clean[tuple([s1,s2])].add(self.NullAction)
    

    def RMP_clean_states_EZ(self,a1):
        """Cleans and updates non-dominated actions for each action in the problem."""
        # Initialize defaultdicts once per action
        all_candid_head_given_tail = defaultdict(set)
        all_candid_tail_given_head = defaultdict(set)

        my_head, my_tail = a1.node_head, a1.node_tail
        head_in_prob, tail_in_prob = self.resStates_minus_by_node[my_head], self.resStates_minus_by_node[my_tail]

        # Process tail states
        action_ub_tail_head = self.my_Multi_Graph_Object.action_ub_tail_head[a1]
        for s_tail in action_ub_tail_head & tail_in_prob:
            all_heads = action_ub_tail_head[s_tail] & head_in_prob
            do_remove = self.state_max_dom_dict| all_heads
            all_candid_head_given_tail[s_tail] = all_heads - do_remove

        # Process head states
        action_ub_head_tail = self.my_Multi_Graph_Object.action_ub_head_tail[a1]
        for s_head in action_ub_head_tail & head_in_prob:
            all_tails = action_ub_head_tail[s_head] & tail_in_prob
            do_remove = self.state_min_dom_dict| all_tails
            all_candid_tail_given_head[s_head] = all_tails - do_remove

            # TODO:Compute tails to connect
            tails_to_connect = subset_where_z_in_Y(s_head, all_candid_tail_given_head[s_head], all_candid_head_given_tail)
            for s_tail in tails_to_connect:
                self.actions_s1_s2_non_dom[(s_tail, s_head)].add(a1)

    def RMP_make_equiv_classes(self):
        #make all equivelence classes
        self.equiv_class_2_s1_s2_pairs=dict() #this will map a number to the s1,s2 pairs that have common action sets 
        self.equiv_class_2_actions=dict()
        for [s1,s2] in self.actions_s1_s2_clean: #iterate over s1,s2
            my_list=[s1.node,s2.node] #create object to store action ids 
            for a in self.actions_s1_s2_clean[tuple([s1,s2])]: #store all action ids
                my_list.append(a.action_id)
            my_list=sorted(my_list) #sort the actions ids
            my_list=str(my_list) #convert the action ids to a string
            self.equiv_class_2_s1_s2_pairs[my_list].add(tuple([s1,s2])) #add the new edge to the equivlenece clas
            if my_list not in self.equiv_class_2_actions:
                self.equiv_class_2_actions[my_list]=self.actions_s1_s2_clean[tuple([s1,s2])]