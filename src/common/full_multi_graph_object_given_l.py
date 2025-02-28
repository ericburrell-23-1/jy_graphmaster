from collections import defaultdict
from typing import Any, Dict, Tuple, Set, DefaultDict, List
import networkx as nx
from itertools import permutations
from src.common.helper import Helper
import numpy as np
from src.common.action import Action
from src.common.state import State
from src.common.helper import Helper
class Full_Multi_Graph_Object_given_l:
 
    #Computed once multi-graph which is generated once
    def __init__(self, l_id, res_states:set[State], all_actions: set[Action], dom_actions_pairs,the_null_action):
        """Initializes the object with states, actions, and null action setup."""
        self.l_id = l_id  # ID for the l ∈ Ω_R generating this
        self.res_states = res_states  # set of all states
        for state in self.res_states:
            print(state.node, state.state_vec.toarray())
        self.all_actions = all_actions  # Set of all possible actions (excluding null action)
        self.dom_actions_pairs = dom_actions_pairs  # Dominating action pairs dictionary
       # self.null_action_info = null_action_info
        self.null_action = the_null_action
        #self.resource_name_to_index = resource_name_to_index
        #self.number_of_resources = number_of_resources
        #self.nullAction = self.make_null_action(size_rhs, number_of_resources)  # Create and assign null action
 
        # Initialize dictionary grouping states by node
        self.resStates_by_node:DefaultDict[int,Set[State]] = defaultdict(set)
        for s in res_states:
            self.resStates_by_node[s.node].add(s)  # Append the actual state object
 
        # Optimized check for source and sink nodes
        node_states = self.resStates_by_node  # Store dictionary lookup once
        source_count = len(node_states.get(-1, []))
        sink_count = len(node_states.get(-2, []))
 
        if source_count != 1 or sink_count != 1:
            raise ValueError(
                f"Graph {l_id} must have exactly one source and one sink, "
                f"but found {source_count} source(s) and {sink_count} sink(s)."
            )
        
    
    def make_state_id_to_state(self):
        """Creates a mapping from state ID to state object."""
        
        self.state_id_to_state = {
            my_state.state_id: my_state
            for node in self.resStates_by_node
            for my_state in self.resStates_by_node[node]
        }
 
    def initialize_system(self):
        self.make_state_id_to_state()
        self.compute_actions_ub()
        self.compute_dom_states_by_node()
        self.PGM_sub_compute_min_dominating_states_by_node()
        self.PGM_sub_compute_maximum_dominated_states_by_node()
 
        self.PGM_clean_states_EZ()
        #self.PGM_make_null_actions()
        self.PGM_compute_remove_redundant_actions()
        self.PGM_make_equiv_classes()
        self.construct_pricing_pgm_graph()
 
    def compute_actions_ub(self):
        """Computes upper bound actions for each (s1, s2) pair."""
        
        # Initialize defaultdicts properly
        self.actions_ub_given_s1s2_2 = defaultdict(set)
        self.action_tail_head = defaultdict(set)
        self.actions_head_tail = defaultdict(set)
        self.action_ub_tail_head = defaultdict(lambda: defaultdict(set))
        self.action_ub_head_tail = defaultdict(lambda: defaultdict(set))
 
        # Iterate over all actions
        for a1 in self.all_actions:
            node_tail, node_head = a1.node_tail, a1.node_head
            #print('a1')
            #print(a1.action_id)
            #print('node_tail, node_head')
            #print([node_tail, node_head])
            # Process state pairs
            #print('resStates_by_node[node_tail]')
            #print(self.resStates_by_node[node_tail])
            for state_tail in self.resStates_by_node[node_tail]:
                head_ideal = a1.get_head_state(state_tail,self.l_id)
                #if node_tail==-1:
                #    print('head_ideal')
                #    print(head_ideal)
                #    print('node_head')
                #    print(node_head)
                #    print('head_ideal==None')
                #    print(head_ideal==None)
                #    print('self.resStates_by_node[node_head]')
                #    print(self.resStates_by_node[node_head])
                #    input('hi')
                #print('hi')
                if head_ideal== None:
                    continue
                for state_head in self.resStates_by_node[node_head]:
                    
                    does_dom,does_equal=head_ideal.this_state_dominates_input_state(state_head)
                    #if node_tail==-1:
                    #    print('a1.resource_consumption_vec')
                    #    print(a1.resource_consumption_vec)
                    #    print('head_ideal.state_vec')
                    #    print(head_ideal.state_vec)
                    #    print('state_head.state_vec')
                    #    print(state_head.state_vec)
                    #    print('does_dom')
                    #    print(does_dom)
                    #    print('does_equal')
                    #    print(does_equal)
                    #    print('node_tail')
                    #    print(node_tail)
                    #    print('node_head')
                    #    print(node_head)
                    #    input('im here this is good')
                    if does_dom or does_equal: #head_ideal.this_state_dominates_input_state(state_head): #check if the ideal head dominates the candidate
                        key = (state_tail, state_head)
 
                        # Store results efficiently
                        self.actions_ub_given_s1s2_2[key].add(a1)
                        self.action_ub_tail_head[a1][state_tail].add(state_head)
                        self.action_ub_head_tail[a1][state_head].add(state_tail)
                        #if node_tail==-1:
                        #    print('im here this is what I wanted')
                        #    print('node_head')
                        #    print(node_head)
                        #    print('node_tail')
                        #    print(node_tail)
                        #    input('hold')
            #if node_tail==-1:
            #    print('node_head')
            #    print(node_head)
            #    print('node_tail')
            #    print(node_tail)
            #    print('input hi')
        #print('dojne making actions ub')
        #input('----')
    def compute_dom_states_by_node(self):
        #Creates two objects that will be key in the rest of the document
        #state_2_dom_states_dict is a dictionary that when s is put in provdies all states taht s dominates
        #  MEANING s1 in state_2_dom_states_dict:  IFF s1<s
        #state_2_is_dom_states_dict is a dictionary that when s is put in provdies all states that dominate s
        #  MEANING s2 in state_2_dom_states_dict:  IFF s1>s
 
        self.state_2_dom_states_dict = defaultdict(set)
        self.state_2_is_dom_states_dict = defaultdict(set)
    
    # Iterate over all nodes and compute dominance
        for my_node, states in self.resStates_by_node.items():
            for s1, s2 in permutations(states, 2):  # Generate all ordered pairs (s1, s2)
                if s1.this_state_dominates_input_state(s2):
                    self.state_2_dom_states_dict[s1].add(s2)
                    self.state_2_is_dom_states_dict[s2].add(s1)
    def PGM_sub_compute_min_dominating_states_by_node(self):
        #Compute for each s the minimally dominating states .
        #s1 in self.state_min_dom_dict[s] meaning s1>s
        #iff s1 in self.state_2_is_dom_states_dict[s] and no s2 exists s.t.
            #s1 in self.state_2_is_dom_states_dict[s] and s2 in self.state_2_is_dom_states_dict[s1]
            #MENAING  s1>s2 adn s2>s
        self.state_min_dom_dict=dict() #Create a dictionary
        for s in self.state_2_is_dom_states_dict: #itterate over all states
            do_remove=Helper.union_of_sets(self.state_2_is_dom_states_dict,self.state_2_is_dom_states_dict[s]) #compute states to remove
            self.state_min_dom_dict[s]=self.state_2_is_dom_states_dict[s]-do_remove#create object to store states
 
    def PGM_sub_compute_maximum_dominated_states_by_node(self):
        #Compute for each s the maximally dominated states .
 
        #s1 in self.state_max_dom_dict[s]
        #iff s1 in state_2_dom_states_dict and no s2 exists s.t.
            #s2 in self.state_2_dom_states_dict[s] and s1 in self.state_2_dom_states_dict[s2]
            #MENAING  s2<s1 adn s2>s
        self.state_max_dom_dict=dict()#Crate place to store maximally dominated stats
        
        for s in self.state_2_dom_states_dict: #iterate over all states s
            do_remove=Helper.union_of_sets(self.state_2_dom_states_dict,self.state_2_dom_states_dict[s]) #compute states to remove
            self.state_max_dom_dict[s]=self.state_2_dom_states_dict[s]-do_remove#create object to store states
 
    
    def PGM_clean_states_EZ(self):
        #go through all of the states and make srue that only symetrically non-dominated actions are included
        #see the rmp version for details
        
        self.actions_s1_s2_non_dom=defaultdict(set)
 
        for a1 in self.all_actions:
            all_candid_head_given_tail=defaultdict(set)
            all_candid_tail_given_head=defaultdict(set)
            for s_tail in self.action_ub_tail_head[a1]:
                all_heads=self.action_ub_tail_head[a1][s_tail]
                do_remove=Helper.union_of_sets(self.state_max_dom_dict,all_heads)
                all_candid_head_given_tail[s_tail]=all_heads-do_remove
            for s_head in self.action_ub_head_tail[a1]:
                all_tails=self.action_ub_head_tail[a1][s_head]
                do_remove=Helper.union_of_sets(self.state_min_dom_dict,all_tails)
                all_candid_tail_given_head[s_head]=all_tails-do_remove
                tails_to_connect=Helper.subset_where_z_in_Y(s_head,all_candid_tail_given_head[s_head],all_candid_head_given_tail)
                for s_tail in tails_to_connect:
                    self.actions_s1_s2_non_dom[(s_tail, s_head)].add(a1)
 
    
    def PGM_compute_remove_redundant_actions(self):
        #remove any dominated actions  from each s1,s2
        #see teh rmp vesion for detailss
        self.actions_s1_s2_clean:Dict[Tuple[State,State],Set[Action]] = defaultdict(set)
        for [s1,s2] in self.actions_s1_s2_non_dom: #iterate over non-full s1,s2
            my_tup=(s1,s2)
            my_actions=self.actions_s1_s2_non_dom[my_tup] #grab the actions
            do_remove=Helper.union_of_sets(self.dom_actions_pairs,my_actions)
            self.actions_s1_s2_clean[my_tup]=my_actions-do_remove
 
    #def PGM_make_null_actions(self):  
        #makes null action terms.  This is for dropping resources
        #see the RMP version for this
     #   for s1 in self.state_max_dom_dict:
     #       for s2 in self.state_max_dom_dict[s1]:
     #           this_null_action = Action(self.null_action_info['trans_min_input'],
     #                                     self.null_action_info['trans_term_add'],self.null_action_info['trans_term_min'],
     #                                     s2,s1,self.null_action_info['contribution_vector'],self.null_action_info['cost'],
     #                                     self.null_action_info['min_resource_vec'],self.null_action_info['resource_consumption_vec'],
     #                                     self.null_action_info['indices_non_zero_max'],self.null_action_info['max_resource_vec'])
     #           self.actions_s1_s2_clean[(s1,s2)].add(this_null_action)
     #           self.null_action.add(this_null_action)
 
    def PGM_make_equiv_classes(self):
        #make all equivelence classes
        #see the RMP version for details
        self.equiv_class_2_s1_s2_pairs: DefaultDict[str, set[Tuple[State, State]]] = defaultdict(set) #this will map a number to the s1,s2 pairs that have common action sets
        self.equiv_class_2_actions:DefaultDict[str,Set[Action]]=defaultdict(set) #this will map a number to the s1,s2 pairs that have common action sets
        for [s1,s2] in self.actions_s1_s2_clean: #iterate over s1,s2
            #my_list=[s1.node,s2.node] #create object to store action ids
            my_name_id=[s1.node,s2.node] #create object to store action ids
            my_action_list = []
            for a in self.actions_s1_s2_clean[(s1,s2)]: #store all action ids
                my_action_list.append(a.action_id)
            my_action_list=sorted(my_action_list) #sort the actions ids
            my_name_id.extend(my_action_list)
            my_name_id=str(my_name_id) #convert the action ids to a string
            self.equiv_class_2_s1_s2_pairs[my_name_id].add((s1,s2)) #add the new edge to the equivlenece clas
            if my_name_id not in self.equiv_class_2_actions:
                self.equiv_class_2_actions[my_name_id]=self.actions_s1_s2_clean[(s1,s2)]
    def PGM_equiv_class_dual_2_low(self, dual_exog_vec):
        """Computes the lowest reduced cost action per equivalence class."""
        
        # Compute reduced costs for all actions

        self.action_2_red_cost = {a1: a1.comp_red_cost(dual_exog_vec) for a1 in self.all_actions}
 
        # Find the action with the lowest reduced cost per equivalence class
        self.equiv_class_2_low_red_action = {}
 
        for my_eq_class in self.equiv_class_2_actions:#iterate overs all equivelnce classes
            min_a1 = min(self.equiv_class_2_actions[my_eq_class], key=lambda a1: self.action_2_red_cost[a1])#copute loewst reduced cost action
            self.equiv_class_2_low_red_action[my_eq_class] = (min_a1, self.action_2_red_cost[min_a1])# compute the lowest reduced cost action and store the reduced cost
 
    def construct_pricing_pgm_graph(self):
        """Constructs the PGM graph with (state_id_tail, state_id_head, equiv_class_id) tuples."""
    
        self.my_rows_pgm_pricing = [
            (s1, s2, eq)
            for eq, pairs in self.equiv_class_2_s1_s2_pairs.items()
            for s1, s2 in pairs
        ]
 
 
    def construct_specific_pricing_pgm(self, dual_exog_vec,rezStates_minus_by_node):
        """Constructs the PGM pricing graph, computes the shortest path, and extracts the ordered list of rows used."""
        
        # Step 1: Compute reduced costs and construct the pricing graph rows
        self.PGM_equiv_class_dual_2_low(dual_exog_vec)
 
        self.rows_pgm_spec_pricing = [
            (row[0].state_id, row[1].state_id, eq_class, action_red_cost, action)
            for row in self.my_rows_pgm_pricing
            for eq_class in [row[2]]  # Extract eq_class cleanly
            for action, action_red_cost in [self.equiv_class_2_low_red_action[eq_class]]  # Unpack action tuple
        ]
 
        # Step 2: Create directed graph
        self.pgm_graph = nx.DiGraph()
        #TODO:
        # Step 3: Add edges (tail -> head) with weights (4th index = action_red_cost)
        #print('making graph')
        for tail, head, _, action_red_cost, action in self.rows_pgm_spec_pricing:
            self.pgm_graph.add_edge(tail,head,  weight=action_red_cost, action=action)
        #    print(f'node_head:{action.node_head}-{head},node_tail:{action.node_tail}-{tail},weight:{action_red_cost}')
        #    print(f'node_head:{action.node_tail},node_tail:{action.node_head},weight:{action_red_cost}')
        #print('check here')
        #input('----')
        # Step 4: Compute the shortest path from source to sink
        # shortest_path = nx.shortest_path(self.pgm_graph, source=rezStates_minus_by_node[-1].state_id, target=rezStates_minus_by_node[-2].state_id, weight="weight", method="dijkstra")
 
        # # Compute the shortest path cost
        # shortest_path_length = nx.shortest_path_length(self.pgm_graph, source=-1, target=-2, weight="weight", method="dijkstra")
        shortest_path_length, shortest_path = nx.single_source_dijkstra(self.pgm_graph,
                                                          source=rezStates_minus_by_node[self.l_id][-1][0].state_id,
                                                            target=rezStates_minus_by_node[self.l_id][-2][0].state_id,
                                                            weight="weight"
                                                        )
        # Step 5: Extract the ordered list of states and actions along the shortest path
        ordered_path_rows = [
            (tail, head, self.pgm_graph[tail][head]["action"])
            for tail, head in zip(shortest_path[:-1], shortest_path[1:])
        ]
        
        return shortest_path, shortest_path_length, ordered_path_rows
    # def make_null_action(self, size_rhs, size_res_vec):
    #     """Creates a NullAction with zero transitions and no exogenous contribution."""
    #     trans_min_input = np.zeros(size_res_vec)  # Minimum input term
    #     trans_term_add = np.zeros(size_res_vec)  # Addition term
    #     trans_term_min = np.full(size_res_vec, np.inf)  # Minimum transition term
    #     node_tail, node_head = None, None  # No tail or head for null action
    #     action_id = "NullAction"  # Unique identifier for the null action
    #     Exog_vec = np.zeros(size_rhs)  # Exogenous contribution vector
    #     cost = 0  # Null action has no cost
    #     non_zero_indices_exog = []  # Empty since Exog_vec is all zeros
    #     min_resource_vec = np.zeros(size_res_vec)
    #     resource_consumption_vec = np.zeros(size_res_vec)
    #     indices_non_zero_max = []    
    #     max_resource_vec = np.full(size_res_vec, np.inf)
    #     #indices_non_zero_max,max_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,trans_term_min)
    #     return Action(trans_min_input, trans_term_add, trans_term_min, node_tail, node_head, Exog_vec, cost, min_resource_vec,resource_consumption_vec,indices_non_zero_max,max_resource_vec )