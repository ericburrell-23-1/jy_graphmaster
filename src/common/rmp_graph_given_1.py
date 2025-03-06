from collections import defaultdict
from typing import Any, Dict, Tuple, Set, DefaultDict
from src.common.helper import Helper
from src.common.action import Action
from src.common.state import State
from src.common.full_multi_graph_object_given_l import Full_Multi_Graph_Object_given_l
from uuid import UUID
from scipy import sparse
class RMP_graph_given_l:

    

    def __init__(self,my_Multi_Graph_Object,resStates_minus_by_node:Dict[int,Set[State]],res_actions,dominated_action,the_null_action,action_id_2_actions):
        self.my_Multi_Graph_Object: Full_Multi_Graph_Object_given_l =my_Multi_Graph_Object#provides the multigrpah object for the l\in Omega_R generating this. 
        self.resStates_minus_by_node=resStates_minus_by_node #dictionary that taks in the node and returns all states assocaited with that node in ResStates minus
        self.res_actions=res_actions #set that holds all actions that are possible.  DOES NOT Includes the Null action
        self.nodes=self.resStates_minus_by_node.keys()#grabs all nodes by grabbing the keys
        self.nullAction_info=the_null_action # null action
        self.dom_actions_pairs = dominated_action
        self.l_id=self.my_Multi_Graph_Object.l_id
        self.action_id_2_actions=action_id_2_actions
        self.debug_check_all_states_unique_minus_id()
        self.debug_check_all_states_in_parent()
        self.debug_check_single_source_sink()
    
    def debug_action_ub(self):
        # remove dominated action for state pair
        def is_elementwise_greater_equal(matrix1, matrix2):
            # Convert to dense if matrices are small enough, or use the approach below for larger matrices
            if matrix1.shape != matrix2.shape:
                return False
            
            # Check if matrix1 - matrix2 has any negative elements
            diff = matrix1 - matrix2
            
            # Get minimum value in the difference matrix
            min_value = diff.data.min() if diff.nnz > 0 else 0
            
            # If minimum value is >= 0, then matrix1 is element-wise >= matrix2
            return min_value >= 0
        def element_wise_minimum(matrix1, matrix2):
            if matrix1.shape != matrix2.shape:
                raise ValueError("Matrices must have the same shape")
            
            # Convert to COO format for easier manipulation
            A = matrix1.tocoo()
            B = matrix2.tocoo()
            
            # Create dictionaries to store values from both matrices
            values_A = {(i, j): v for i, j, v in zip(A.row, A.col, A.data)}
            values_B = {(i, j): v for i, j, v in zip(B.row, B.col, B.data)}
            
            # Get all positions where either matrix has a non-zero value
            all_positions = set(values_A.keys()) | set(values_B.keys())
            
            # Build the result matrix
            rows, cols, data = [], [], []
            for i, j in all_positions:
                # Get values, defaulting to 0 if position not in matrix
                val_A = values_A.get((i, j), 0)
                val_B = values_B.get((i, j), 0)
                
                # Take the minimum value
                min_val = min(val_A, val_B)
                
                # Only add non-zero values to keep sparsity
                if min_val != 0:
                    rows.append(i)
                    cols.append(j)
                    data.append(min_val)
            
            # Create a new CSR matrix with the minimum values
            result = sparse.csr_matrix((data, (rows, cols)), shape=matrix1.shape)
            
            return result
        def is_elementwise_strictly_greater(matrix1, matrix2):
            """
            Check if matrix1 is element-wise strictly greater than matrix2.
            Returns True if all elements in matrix1 > corresponding elements in matrix2.
            
            Parameters:
            -----------
            matrix1, matrix2 : scipy.sparse.csr_matrix
                Sparse matrices to compare
            
            Returns:
            --------
            bool
                True if matrix1 > matrix2 element-wise, False otherwise
            """
            if matrix1.shape != matrix2.shape:
                return False
            
            # Convert matrices to COO format for easier element access
            A = matrix1.tocoo()
            B = matrix2.tocoo()
            
            # Create dictionaries of non-zero elements
            values_A = {(i, j): v for i, j, v in zip(A.row, A.col, A.data)}
            values_B = {(i, j): v for i, j, v in zip(B.row, B.col, B.data)}
            
            # Get all positions where either matrix has a value
            all_positions = set(values_A.keys()) | set(values_B.keys())
            
            for i, j in all_positions:
                # Get values, defaulting to 0 if position not in matrix
                val_A = values_A.get((i, j), 0)
                val_B = values_B.get((i, j), 0)
                
                # If any element in A is not strictly greater than in B, return False
                if val_A <= val_B:
                    return False
            
            return True
        this_action_dict = defaultdict(list)
        for action in self.res_actions:
            this_action_dict[(action.node_tail,action.node_head)].append(action)
        
        for key, action_list in this_action_dict.items():
            remove_action_index = []
            for a1_idx in range(1,len(action_list)):
                for a2_idx in range(len(action_list)-1):
                    if action_list[a1_idx].get_is_dominated(action_list[a2_idx]):
                        remove_action_index.append(a2_idx)
                    if  action_list[a2_idx].get_is_dominated(action_list[a1_idx]):
                        remove_action_index.append(a1_idx)
            this_action_dict[key] = [action_list[i] for i in range(len(action_list)) if i not in remove_action_index]
        # remove dominated states
        this_res_state = set()
        for node, states in self.resStates_minus_by_node.items():
            this_res_state.update(states)

        # get action ub
        action_ub = defaultdict(set)
        for s1 in this_res_state:
            for s2 in this_res_state:
                if s2 != s1:
                    if s1.node != s2.node :
                        if (s1.node,s2.node) in this_action_dict:
                            for action in this_action_dict[(s1.node,s2.node)]:
                                if is_elementwise_greater_equal(s1.state_vec,action.min_resource_vec) and \
                                    is_elementwise_greater_equal(element_wise_minimum(s1.state_vec + action.resource_consumption_vec,action.max_resource_vec),s2.state_vec):
                                    action_ub[(s1,s2)].add(action)
                    else:
                        if is_elementwise_strictly_greater(s1.state_vec,s2.state_vec):
                            action_ub[(s1,s2)].add(self.null_action)

        for (s1,s2) in action_ub.keys():
            if s2 == s1:
                input('key error here')
        # action clean
        for(s1,s2), action_list2 in action_ub.items():
            remove_action = set()
            for action in action_list2:
                for (s31,s32), action_list3 in action_ub.items():
                    if s1 == s31 and s2 != s32:
                        is_dom, is_equal = s32.this_state_dominates_input_state(s2)
                        if is_dom and action in action_list3:
                            remove_action.add(action)
                    if s1 != s31 and s2 == s32:
                        is_dom, is_equal = s1.this_state_dominates_input_state(s31)
                        if is_dom and action in action_list3:
                            remove_action.add(action)
                for action2 in action_list2:
                    if action2.get_is_dominated(action) == True:
                        remove_action.add(action)
            action_list2.difference_update(remove_action)
        print('check action ub here')
        # for (s1,s2) ,action_list in action_ub.items():
        #     if (s1,s2) not in self.actions_ub_given_s1s2_2.keys() and len(action_list)>0:
        #         input('key not exsist ')
        #     for action in action_list:
        #         if action not in self.actions_ub_given_s1s2_2[(s1,s2)]:
        #             input(f'action error')
        # Check action_ub against self.actions_ub_given_s1s2_2
        #self.actions_ub_given_s1s2_2
        
        for (s1, s2), action_list in action_ub.items():
            if len(action_list) > 0:
                if (s1, s2) not in self.actions_s1_s2_clean:
                    print(f"Key {(s1, s2)} exists in action_ub but not in self.actions_ub_given_s1s2_2")
                else:
                    missing_actions = [action for action in action_list if action not in self.actions_s1_s2_clean[(s1, s2)]]
                    if missing_actions:
                        print(f"Missing actions for key {(s1, s2)}: {missing_actions}")

        # Check self.actions_ub_given_s1s2_2 against action_ub
        for (s1, s2), action_list in self.actions_s1_s2_clean.items():
            if len(action_list) > 0:
                if (s1, s2) not in action_ub:
                    print(f"Key {(s1, s2)} exists in self.actions_ub_given_s1s2_2 but not in action_ub")
                else:
                    missing_actions = [action for action in action_list if action not in action_ub[(s1, s2)]]
                    if missing_actions:
                        print(f"Missing actions for key {(s1, s2)}: {missing_actions}")
        
        print('check action ub end')

    def debug_check_all_states_unique_minus_id(self):
        my_states_all=[]
        for n in self.resStates_minus_by_node:
            for s in self.resStates_minus_by_node[n]:
                my_states_all.append(s)
        for i in range(0,len(my_states_all)):
            for j in range(0,len(my_states_all)):
                if i!=j:
                    s1=my_states_all[i]
                    s2=my_states_all[j]
                    if s1.equals_minus_id(s2):
                        print('---')
                        s1.pretty_print_state()
                        s2.pretty_print_state()
                        input('error here s1=s2')
    def  debug_check_all_states_in_parent(self):
        for n in self.resStates_minus_by_node:
            for s in self.resStates_minus_by_node[n]:
                if s not in self.my_Multi_Graph_Object.rez_states:
                    print('error here not in paretn')
                    print('l_id')
                    print(self.l_id)
                    print('---')
                    s.pretty_print_state()
                    print('---')
                    print('self.my_Multi_Graph_Object.source_state.state_id')
                    print(self.my_Multi_Graph_Object.source_state.state_id)
                    print('self.my_Multi_Graph_Object.sink_state.state_id')
                    print(self.my_Multi_Graph_Object.sink_state.state_id)
                    input('----')
       
    def debug_check_single_source_sink(self):

        if len(self.resStates_minus_by_node[-1])!=1 or len(self.resStates_minus_by_node[-2])!=1:

            print('type(resStates_minus_by_node)')
            print(type(self.resStates_minus_by_node))
            print('self.resStates_minus_by_node[-1]')
            print(self.resStates_minus_by_node[-1])
            print('self.resStates_minus_by_node[-2]')
            print(self.resStates_minus_by_node[-2])
            print('we got two here')
            input('---')
            for s2 in self.resStates_minus_by_node[-2]:
                s2.pretty_print_state()
            for s1 in self.resStates_minus_by_node[-1]:
                s1.pretty_print_state()
            input('---')
        

        node_states = self.resStates_minus_by_node  # Store dictionary lookup once
        source_count = len(node_states.get(-1, []))
        sink_count = len(node_states.get(-2, []))
 
        if source_count != 1 or sink_count != 1:
            raise ValueError(
                f"RMP Graph {self.l_id} must have exactly one source and one sink, "
                f"but found {source_count} source(s) and {sink_count} sink(s)."
            ) 

    def make_state_id_2_state(self):
        self.state_id_to_state=dict()

        if len(self.nodes) == 0:
            raise ValueError("nodes is empty. ")
        if len(self.res_actions) == 0:
            raise ValueError("res actions is empty. ")
        for my_node in self.resStates_minus_by_node:
            for my_state in self.resStates_minus_by_node[my_node]:
                self.state_id_to_state[my_state.state_id] = my_state
                if my_state.l_id!=self.l_id:
                    print('my_state.l_id')
                    print(my_state.l_id)
                    print('self.l_id')
                    print(self.l_id)
                    input('errror here')

    def debug_check_actions_ok(self):
        for a1 in self.res_actions:
            self.my_Multi_Graph_Object.action_ub_tail_head[a1]
            for s_tail in  self.my_Multi_Graph_Object.action_ub_tail_head[a1]:
                for s_head in  self.my_Multi_Graph_Object.action_ub_tail_head[a1][s_tail]:
                    a1.check_valid(s_tail, s_head)
        #input('pre clearence check done')

    def initialize_system(self):
        #initialize teh RMP graph
        
        self.make_state_id_2_state()
        for u in self.nodes:#self.my_Multi_Graph_Object.resStates_minus_by_node:
            self.update_domination(u) #grab domination list and dominated list for each u
            self.RMP_update_sub_compute_min_dominating_states_by_node(u)#update minimum domination dictionary
            self.RMP_update_sub_compute_maximum_dominated_states_by_node(u)#update maximum dominated dictionary
        self.actions_s1_s2_non_dom = defaultdict(set) #create an object called actions dominated 
        
        debug_seen_actions=set([])
        self.res_actions=list(set(self.res_actions))
        self.debug_check_actions_ok()
        
        for a1 in self.res_actions:# iterate over actions and comptue the a set that is actions NOT actions ub; except that we may have dominated actions which will be removed next
           #a1=self.action_id_2_actions[a1_id]
           if type(a1)!= Action:
               print('a1')
               print(a1)
               input('error this needs to be an action')
           self.RMP_clean_states_EZ(a1)
           #self.DEBUG_RMP_clean_states_EZ(a1)
           #if a1 in self.deb
        self.RMP_make_null_actions()
        self.RMP_compute_remove_redundant_actions()
        self.RMP_make_equiv_classes()
        self.debug_action_ub()
    def update_domination(self, my_node):
        """Updates domination relationships for all states of a given node."""

        # Ensure defaultdict(set) for automatic key initialization
        self.all_included_states_dominating_s = defaultdict(set)
        self.all_included_states_that_s_dominates = defaultdict(set)

        all_states = self.resStates_minus_by_node[my_node]  # Avoid redundant dictionary lookup

        #for each state[s] we compute all states that dominate s and that s dominates by having the multigraph already computed which just grab the relavant terms
        for s in all_states:
            self.all_included_states_dominating_s[s] = self.my_Multi_Graph_Object.state_2_is_dom_states_dict[s].intersection(all_states)
            self.all_included_states_that_s_dominates[s] = self.my_Multi_Graph_Object.state_2_dom_states_dict[s].intersection(all_states)


    def RMP_update_sub_compute_maximum_dominated_states_by_node(self,my_node):
        #compute the maximum dominated state dictionary.  
        self.state_max_dom_dict:Dict[State,Set[State]] = dict()#Crate place to store maximally dominated stats 
        
        for s in self.resStates_minus_by_node[my_node]: #iterate over all states s
            #compute all states that are dominated by a sets that s dominates.  these are gonna be removed
            do_remove=Helper.union_of_sets(self.all_included_states_that_s_dominates,self.all_included_states_that_s_dominates[s])
            self.state_max_dom_dict[s]=self.all_included_states_that_s_dominates[s]-do_remove#create object to store states

    def RMP_update_sub_compute_min_dominating_states_by_node(self, my_node):
    
        self.state_min_dom_dict = {}  # Create dictionary to store minimally dominated states
        
        #for s in self.all_states_of_node[my_node]:  # Iterate over all states in the node
        for s in self.resStates_minus_by_node[my_node]:
            #compute all states that are dominated  by a states that are dominated by s.  these are gonna be removed

            do_remove = Helper.union_of_sets(self.all_included_states_dominating_s, self.all_included_states_dominating_s[s] )
            self.state_min_dom_dict[s] = self.all_included_states_dominating_s[s] - do_remove  # Store minimally dominated states

    def RMP_compute_remove_redundant_actions(self):
        """Removes dominated actions from each (s1, s2) pair."""
        
        self.actions_s1_s2_clean:Dict[Tuple[State,State],Set[Action]] = defaultdict(set)

        #for s1, s2 in self.actions_s1_s2:  # Iterate over (s1, s2) pairs
        for s1, s2 in self.actions_s1_s2_non_dom:

            if s1.state_id not in self.state_id_to_state:
                input('error here should be ')
            if s2.state_id not in self.state_id_to_state:
                input('error here should be 2')
            my_tup = (s1, s2)
            my_actions = self.actions_s1_s2_non_dom[my_tup]  # Get non-dominated actions
            do_remove = Helper.union_of_sets(self.dom_actions_pairs, my_actions)
            
            # Store cleaned actions (removing dominated ones)
            self.actions_s1_s2_clean[my_tup] = my_actions - do_remove

    def RMP_make_null_actions(self):  
        #makes null action terms.  This is for dropping resources 
        for s1 in self.state_max_dom_dict:
            for s2 in self.state_max_dom_dict[s1]:
                self.actions_s1_s2_non_dom[(s1,s2)].add(self.my_Multi_Graph_Object.null_action)
    
    
    def debug_check_that_res_minus_states_agree(self):

        for my_node in self.resStates_minus_by_node:
            for my_state in self.resStates_minus_by_node[my_node]:
                if my_state.state_id not in  self.state_id_to_state:
                    input('error here no coresp')

    def debug_check_state_is_in_states_id(self,my_state):

        if my_state.state_id not in self.state_id_to_state:
            print('not found')
            my_state.pretty_print_state()
            input('error here')


    def DEBUG_RMP_clean_states_EZ(self,a1:Action):
        """Cleans and updates non-dominated actions for each action in the problem."""
        # Initialize defaultdicts once per action
        my_head_node= a1.node_head
        my_tail_node = a1.node_tail # grab the head adn the tail
        #grab teh states corresponding to the head and the tail
        self.debug_check_that_res_minus_states_agree()
        head_in_prob= list(self.resStates_minus_by_node[my_head_node])
        tail_in_prob = list(self.resStates_minus_by_node[my_tail_node])
        self.debug_check_that_res_minus_states_agree()
        
        action_ub_tail_head = self.my_Multi_Graph_Object.action_ub_tail_head[a1] #get all tail to head for this action

       
        # Process tail states
        for s_tail in  action_ub_tail_head: #iterate over all tails  that can start the action and in the poblem 
            if s_tail in tail_in_prob:
                for s_head in  action_ub_tail_head[s_tail]:
                    if s_head in head_in_prob: 
                        #self.debug_check_that_res_minus_states_agree()
                        #self.debug_check_state_is_in_states_id(s_head)
                        a1.check_valid(s_tail, s_head)
                        self.actions_s1_s2_non_dom[(s_tail, s_head)].add(a1)
        

    def RMP_clean_states_EZ(self,a1:Action):
        """Cleans and updates non-dominated actions for each action in the problem."""
        # Initialize defaultdicts once per action
        all_candid_head_given_tail = defaultdict(set) #given teh action and tail these are candidate heads but not not be maximal and hence not included
        all_candid_tail_given_head = defaultdict(set)#given teh action and head these are candidate tails but not not be minmal and hence not included
        my_head, my_tail = a1.node_head, a1.node_tail # grab the head adn the tail
        #grab teh states corresponding to the head and the tail
        head_in_prob, tail_in_prob = self.resStates_minus_by_node[my_head], self.resStates_minus_by_node[my_tail]
        set_head_in_prob=set(head_in_prob)
        set_tail_in_prob=set(tail_in_prob)
        # Process tail states
        debug_all_adds=[]

        action_ub_tail_head = self.my_Multi_Graph_Object.action_ub_tail_head[a1] #get all tail to head for this action
        tmp_ub_tail_2_head=set(action_ub_tail_head.keys())
        tmp_ub_tail_2_head=tmp_ub_tail_2_head.intersection(tail_in_prob)
        for s_tail in  tmp_ub_tail_2_head: #iterate over all tails  that can start the action and in the poblem 
            all_heads = action_ub_tail_head[s_tail].intersection(set_head_in_prob) #anyhing in the ptoblem and i  can go to in the orignial multigrpah
            do_remove = Helper.union_of_sets(self.state_max_dom_dict, all_heads) #comptue the heads to remove: which is  any candidate heads taht are dominated by another candidate head 
            all_candid_head_given_tail[s_tail] = all_heads - do_remove# remove  those heads

        # Process head states
        action_ub_head_tail = self.my_Multi_Graph_Object.action_ub_head_tail[a1] #get all head to tail for this action
        tmp_ub_head_2_tail=set(action_ub_head_tail.keys())
        tmp_ub_head_2_tail=tmp_ub_head_2_tail.intersection(head_in_prob)
        for s_head in tmp_ub_head_2_tail:#action_ub_head_tail.intersection(head_in_prob): #get all heads in the problem and can be the product of the action
            all_tails = action_ub_head_tail[s_head].intersection(set_tail_in_prob) #all tails that can be produced
            do_remove = Helper.union_of_sets(self.state_min_dom_dict, all_tails) #removes the dominated ones 
            all_candid_tail_given_head[s_head] = all_tails - do_remove #do remove operation 

        #julian Check

        #for s_head in action_ub_head_tail.intersection(head_in_prob): #get all heads in the problem and can be the product of the action
        for s_head in tmp_ub_head_2_tail: #get all heads in the problem and can be the product of the action
            
            # Compute tails to connect.  not doinated either way
            tails_to_connect = Helper.subset_where_z_in_Y(s_head, all_candid_tail_given_head[s_head], all_candid_head_given_tail)
            for s_tail in tails_to_connect:
                
                if s_head not in action_ub_tail_head[s_tail]:
                    s_head.pretty_print_state()
                    s_tail.pretty_print_state()
                    input('error here this pair not found')
                a1.check_valid(s_tail, s_head)
                #a1.check_valid(s_tail, s_head)
                if  a1 in  self.actions_s1_s2_non_dom[(s_tail, s_head)]:
                    print('debug_all_adds')
                    print(debug_all_adds)
                    print('tails_to_connect')
                    print(tails_to_connect)
                    input('already found in rmp')
                self.actions_s1_s2_non_dom[(s_tail, s_head)].add(a1)
                #debug_all_adds.append((s_tail, s_head))
                #if self.l_id==1:
                #    print('new edge')
                #    s_tail.pretty_print_state()
                #    s_head.pretty_print_state()
                #    input('----')
    def RMP_make_equiv_classes(self):
        #make all equivelence classes
        self.equiv_class_2_s1_s2_pairs:DefaultDict[str,Set[Tuple[State,State]]] = defaultdict(set) #this will map a number to the s1,s2 pairs that have common action sets 
        self.equiv_class_2_actions: DefaultDict[str,Set[Action]]=dict()
        self.s1_s2_pair_2_equiv=dict()
        DEBUG_every_term_own_class=False
        lid_str="lid = "+str(self.l_id)
        for (s1,s2) in self.actions_s1_s2_clean: #iterate over s1,s2
            my_list=[s1.node,s2.node,lid_str] #create object to store action ids 
            if  DEBUG_every_term_own_class==True:
                my_list=[s1.node,s2.node,s1.state_id,s2.state_id,lid_str]
            my_action_list = []
            for a in self.actions_s1_s2_clean[(s1,s2)]: #store all action ids
                my_action_list.append(a.action_id)
            my_action_list=sorted(my_action_list) #sort the actions ids
            my_list.extend(my_action_list)
            my_list=str(my_list) #convert the action ids to a string
            self.equiv_class_2_s1_s2_pairs[my_list].add((s1,s2)) #add the new edge to the equivlenece clas
            if my_list not in self.equiv_class_2_actions: #if not an  equivelence class already then  create teh actions that make it up
                self.equiv_class_2_actions[my_list]=self.actions_s1_s2_clean[(s1,s2)]
            self.s1_s2_pair_2_equiv[(s1,s2)]=my_list
