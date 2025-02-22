from collections import defaultdict
from src.common.helper import Helper
from src.common.action import Action
from src.common.full_multi_graph_object_given_l import Full_Multi_Graph_Object_given_l
class RMP_graph_given_l:

    def __init__(self,my_Multi_Graph_Object,resStates_minus_by_node,res_actions,nullAction_info):
        self.my_Multi_Graph_Object: Full_Multi_Graph_Object_given_l =my_Multi_Graph_Object#provides the multigrpah object for the l\in Omega_R generating this. 
        self.resStates_minus_by_node=resStates_minus_by_node #dictionary that taks in the node and returns all states assocaited with that node in ResStates minus
        self.res_actions=res_actions #set that holds all actions that are possible.  DOES NOT Includes the Null action
        self.nodes=self.resStates_minus_by_node.keys()#grabs all nodes by grabbing the keys
        self.nullAction_info=nullAction_info # null action
    def initialize_system(self):
        #initialize teh RMP graph
        for u in self.nodes:#self.my_Multi_Graph_Object.resStates_minus_by_node:
            self.update_domination(u) #grab domination list and dominated list for each u
            self.RMP_update_sub_compute_min_dominating_states_by_node(u)#update minimum domination dictionary
            self.RMP_update_sub_compute_maximum_dominated_states_by_node(u)#update maximum dominated dictionary
        self.actions_s1_s2_non_dom = defaultdict(set) #create an object called actions dominated 
        for a1 in self.res_actions:# iterate over actions and comptue the a set that is actions NOT actions ub; except that we may have dominated actions which will be removed next
           self.RMP_clean_states_EZ(a1)
        self.RMP_make_null_actions()
        self.RMP_compute_remove_redundant_actions()
        self.RMP_make_equiv_classes()
    def update_domination(self, my_node):
        """Updates domination relationships for all states of a given node."""

        # Ensure defaultdict(set) for automatic key initialization
        self.all_included_states_dominating_s = defaultdict(set)
        self.all_included_states_that_s_dominates = defaultdict(set)

        all_states = self.resStates_minus_by_node[my_node]  # Avoid redundant dictionary lookup

        #for each state[s] we compute all states that dominate s and that s dominates by having the multigraph already computed which just grab the relavant terms
        for s in all_states:
            self.all_included_states_dominating_s[s] = self.my_Multi_Graph_Object.state_2_is_dom_states_dict[s] & all_states
            self.all_included_states_that_s_dominates[s] = self.my_Multi_Graph_Object.state_2_dom_states_dict[s] & all_states


    def RMP_update_sub_compute_maximum_dominated_states_by_node(self,my_node):
        #compute the maximum dominated state dictionary.  
        self.state_max_dom_dict=dict()#Crate place to store maximally dominated stats 
        
        for s in self.resStates_minus_by_node[my_node]: #iterate over all states s
            #compute all states that are dominated by a sets that s dominates.  these are gonna be removed
            do_remove=Helper.Helper.union_of_sets(self.all_included_states_that_s_dominates,self.all_included_states_that_s_dominates[s])
            self.state_max_dom_dict[s]=self.all_included_states_that_s_dominates[s]-do_remove#create object to store states

    def RMP_update_sub_compute_min_dominating_states_by_node(self, my_node):
    
        self.state_min_dom_dict = {}  # Create dictionary to store minimally dominated states
        
        for s in self.all_states_of_node[my_node]:  # Iterate over all states in the node
            #compute all states that are dominated  by a states that are dominated by s.  these are gonna be removed

            do_remove = Helper.union_of_sets(self.all_included_states_dominating_s, self.all_included_states_dominating_s[s] )
            self.state_min_dom_dict[s] = self.all_included_states_dominating_s[s] - do_remove  # Store minimally dominated states

    def RMP_compute_remove_redundant_actions(self):
        """Removes dominated actions from each (s1, s2) pair."""
        
        self.actions_s1_s2_clean = defaultdict(set)

        for s1, s2 in self.actions_s1_s2:  # Iterate over (s1, s2) pairs
            my_tup = (s1, s2)
            my_actions = self.actions_s1_s2_non_dom[my_tup]  # Get non-dominated actions
            do_remove = Helper.union_of_sets(self.dom_actions_pairs, my_actions)
            
            # Store cleaned actions (removing dominated ones)
            self.actions_s1_s2_clean[my_tup] = my_actions - do_remove

    def RMP_make_null_actions(self):  
        #makes null action terms.  This is for dropping resources 
        for s1 in self.state_max_dom_dict:
            for s2 in self.state_max_dom_dict[s1]:
                this_null_action = Action(self.null_action_info['trans_min_input'],
                                          self.null_action_info['trans_term_add'],self.null_action_info['trans_term_min'],
                                          s2,s1,self.null_action_info['contribution_vector'],self.null_action_info['cost'],
                                          self.null_action_info['min_resource_vec'],self.null_action_info['resource_consumption_vec'],
                                          self.null_action_info['indices_non_zero_max'],self.null_action_info['max_resource_vec'])
                self.actions_s1_s2_clean[tuple([s1,s2])].add(this_null_action)
    

    def RMP_clean_states_EZ(self,a1:Action):
        """Cleans and updates non-dominated actions for each action in the problem."""
        # Initialize defaultdicts once per action
        all_candid_head_given_tail = defaultdict(set) #given teh action and tail these are candidate heads but not not be maximal and hence not included
        all_candid_tail_given_head = defaultdict(set)#given teh action and head these are candidate tails but not not be minmal and hence not included

        my_head, my_tail = a1.node_head, a1.node_tail # grab the head adn the tail
        #grab teh states corresponding to the head and the tail
        head_in_prob, tail_in_prob = self.resStates_minus_by_node[my_head], self.resStates_minus_by_node[my_tail]

        # Process tail states
        #TODO: what is supposed to do
        action_ub_tail_head = self.my_Multi_Graph_Object.action_ub_tail_head[a1] #get all tail to head for this action
        for s_tail in  action_ub_tail_head: #iterate over all tails  that can start the action and in the poblem 
            if s_tail in tail_in_prob:
                all_heads = action_ub_tail_head[s_tail] & head_in_prob #anyhing in the ptoblem and i  can go to in the orignial multigrpah
                do_remove = Helper.union_of_sets(self.state_max_dom_dict, all_heads) #comptue the heads to remove: which is  any candidate heads taht are dominated by another candidate head 
                all_candid_head_given_tail[s_tail] = all_heads - do_remove# remove  those heads

        # Process head states
        action_ub_head_tail = self.my_Multi_Graph_Object.action_ub_head_tail[a1] #get all head to tail for this action
        for s_head in action_ub_head_tail: #get all heads in the problem and can be the product of the action
            if s_head in head_in_prob:
                all_tails = action_ub_head_tail[s_head] & tail_in_prob #all tails that can be produced
                do_remove = Helper.union_of_sets(self.state_min_dom_dict, all_tails) #removes the dominated ones 
                all_candid_tail_given_head[s_head] = all_tails - do_remove #do remove operation 

                # Compute tails to connect.  not doinated either way
                tails_to_connect = Helper.subset_where_z_in_Y(s_head, all_candid_tail_given_head[s_head], all_candid_head_given_tail)
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
            if my_list not in self.equiv_class_2_actions: #if not an  equivelence class already then  create teh actions that make it up
                self.equiv_class_2_actions[my_list]=self.actions_s1_s2_clean[tuple([s1,s2])]