from collections import defaultdict
import networkx as nx
from itertools import permutations

class Full_Multi_Graph_Object_given_l:    
    def __init__(self,l_id,resStates_by_node,all_actions,dom_actions_pairs):
        self.l_id=l_id#provides the id for hte l\in Omega_R generating this.  All nodes shoudl have the same id (even if we treate all nodes as being part of the same graph we can use this )
        self.resStates_by_node=resStates_by_node #dictionary that taks in the node and returns all states assocaited with that node in ResStates
        self.all_actions=all_actions #set that holds all actions that are possible.  Includes the Null action
        self.dom_actions_pairs=dom_actions_pairs #dictionary that for each action has the dominating actions#set that contains  a pair of actions a_1,a_2 if a1 dominates a2
    
    def initialize_system(self):

        self.compute_actions_ub()
        self.compute_dom_states_by_node()
        self.PGM_sub_compute_min_dominating_states_by_node()
        self.PGM_sub_compute_maximum_dominated_states_by_node()

        self.PGM_make_null_actions()
        self.PGM.PGM_compute_remove_redundant_actions()
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

            # Process state pairs
            for state_tail in self.resStates_by_node[node_tail]:
                head_ideal = a1.get_head_state(self, state_tail)
                
                for state_head in self.resStates_by_node[node_head]:
                    if head_ideal.this_state_dominates_input_state(state_head):
                        key = (state_tail, state_head)

                        # Store results efficiently
                        self.actions_ub_given_s1s2_2[key].add(a1)
                        self.action_ub_tail_head[a1][state_tail].add(state_head)
                        self.action_ub_head_tail[a1][state_head].add(state_tail)

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
            do_remove={**self.state_2_is_dom_states_dict,**self.state_2_is_dom_states_dict[s]}
            self.state_min_dom_dict[s]=self.state_2_is_dom_states_dict[s]-do_remove#create object to store states

    def PGM_sub_compute_maximum_dominated_states_by_node(self):
        #Compute for each s the maximally dominated states . 

        #s1 in self.state_max_dom_dict[s]
        #iff s1 in state_2_dom_states_dict and no s2 exists s.t. 
            #s2 in self.state_2_dom_states_dict[s] and s1 in self.state_2_dom_states_dict[s2]
            #MENAING  s2<s1 adn s2>s
        self.state_max_dom_dict=dict()#Crate place to store maximally dominated stats 
        
        for s in self.state_2_dom_states_dict: #iterate over all states s
            do_remove={**self.state_2_dom_states_dict,**self.state_2_dom_states_dict[s]}
            self.state_max_dom_dict[s]=self.state_2_dom_states_dict[s]-do_remove#create object to store states

    
    def PGM_clean_states_EZ(self):
        self.actions_s1_s2_non_dom=defaultdict(set([]))

        for a1 in self.all_actions: 
            all_candid_head_given_tail=defaultdict(set([]))
            all_candid_tail_given_head=defaultdict(set([]))
            for s_tail in self.action_ub_tail_head[a1]:
                all_heads=self.action_ub_tail_head[a1][s_tail]
                do_remove={**self.state_max_dom_dict,**all_heads}
                all_candid_head_given_tail[s_tail]=all_heads-do_remove
            for s_head in self.action_ub_head_tail[a1]:
                all_tails=self.action_ub_head_tail[a1][s_head]
                do_remove={**self.state_min_dom_dict,**all_tails}
                all_candid_tail_given_head[s_head]=all_tails-do_remove
                tails_to_connect=self.subset_where_z_in_Y(s_head,all_candid_tail_given_head[s_head],all_candid_head_given_tail)
                for s_tail in tails_to_connect:
                    self.actions_s1_s2_non_dom[(s_tail, s_head)].add(a1)

    
    def PGM_compute_remove_redundant_actions(self):
        #remove any dominated actions  from each s1,s2
        self.actions_s1_s2_clean=defaultdict(set([]))
        for [s1,s2] in self.actions_s1_s2: #iterate over non-full s1,s2
            my_tup=tuple([s1,s2])
            my_actions=self.actions_s1_s2_non_dom[my_tup] #grab the actions 
            do_remove={**self.dom_actions_pairs,**my_actions}
            self.actions_s1_s2_clean=my_actions-do_remove

    def PGM_make_null_actions(self):  
        #makes null action terms.  This is for dropping resources 
        for s1 in self.state_max_dom_dict:
            for s2 in self.state_max_dom_dict[s1]:
                self.actions_s1_s2_clean[tuple([s1,s2])].add(self.NullAction)
    

    def PGM_make_equiv_classes(self):
        #make all equivelence classes
        self.equiv_class_2_s1_s2_pairs=dict() #this will map a number to the s1,s2 pairs that have common action sets 
        self.equiv_class_2_actions=dict() #this will map a number to the s1,s2 pairs that have common action sets 
        for [s1,s2] in self.actions_s1_s2_clean: #iterate over s1,s2
            my_list=[s1.node,s2.node] #create object to store action ids 
            for a in self.actions_s1_s2_clean[tuple([s1,s2])]: #store all action ids
                my_list.append(a.action_id)
            my_list=sorted(my_list) #sort the actions ids
            my_list=str(my_list) #convert the action ids to a string
            self.equiv_class_2_s1_s2_pairs[my_list].add(tuple([s1,s2])) #add the new edge to the equivlenece clas
            if my_list not in self.equiv_class_2_actions:
                self.equiv_class_2_actions[my_list]=self.actions_s1_s2_clean[tuple([s1,s2])]
    def PGM_equiv_class_dual_2_low(self, dual_exog_vec):
        """Computes the lowest reduced cost action per equivalence class."""
        
        # Compute reduced costs for all actions
        self.action_2_red_cost = {a1: a1.comp_red_cost(dual_exog_vec) for a1 in self.all_actions}

        # Find the action with the lowest reduced cost per equivalence class
        self.equiv_class_2_low_red_action = {}

        for my_eq_class in self.equiv_class_2_actions:
            min_a1 = min(self.equiv_class_2_actions[my_eq_class], key=lambda a1: self.action_2_red_cost[a1])
            self.equiv_class_2_low_red_action[my_eq_class] = (min_a1, self.action_2_red_cost[min_a1])

    def construct_pricing_pgm_graph(self):
        """Constructs the PGM graph with (state_id_tail, state_id_head, equiv_class_id) tuples."""
    
        self.my_rows_pgm_pricing = [
            (s1, s2, eq)
            for eq, pairs in self.equiv_class_2_s1_s2_pairs.items()
            for s1, s2 in pairs
        ]


    def construct_specific_pricing_pgm(self, dual_exog_vec):
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

        # Step 3: Add edges (tail -> head) with weights (4th index = action_red_cost)
        for tail, head, _, action_red_cost, action in self.rows_pgm_spec_pricing:
            self.pgm_graph.add_edge(tail, head, weight=action_red_cost, action=action)

        # Step 4: Compute the shortest path from source to sink
        shortest_path = nx.shortest_path(self.pgm_graph, source="source", target="sink", weight="weight", method="dijkstra")

        # Compute the shortest path cost
        shortest_path_length = nx.shortest_path_length(self.pgm_graph, source="source", target="sink", weight="weight", method="dijkstra")

        # Step 5: Extract the ordered list of states and actions along the shortest path
        ordered_path_rows = [
            (tail, head, self.pgm_graph[tail][head]["action"])
            for tail, head in zip(shortest_path[:-1], shortest_path[1:])
        ]

        return shortest_path, shortest_path_length, ordered_path_rows