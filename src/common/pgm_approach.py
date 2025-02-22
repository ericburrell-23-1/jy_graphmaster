from src.common.rmp_graph_given_1 import RMP_graph_given_l
from collections import defaultdict
from src.common.jy_var import jy_var
import numpy as np
import xpress as xp
class PGM_appraoch:
    #This will do the enitre RMP. 

    #This will do the enitre RMP. 

    def __init__(self,my_PGM_graph_list,prob_RHS,rez_states_minus,rez_actions_minus,incumbant_lp,null_action_info):

        self.my_PGM_graph_list=my_PGM_graph_list #list of all of the PGM graphs
        self.prob_RHS=prob_RHS #RHS
        self.rez_states_minus=rez_states_minus #has all of the states in rez states minus by graph . So if I put in a grpah id then i get out the rez states minus to initialize
        self.make_res_states_minus_by_node()
        self.rez_actions_minus=rez_actions_minus #get all actions that are currently under consdieration
        self.incumbant_lp=incumbant_lp# has incumbent LP objective
        self.null_action_info = null_action_info
        #TODO: this function never defined
        #self.init_jy_options() #has teh options
        
    def make_res_states_minus_by_node(self):
        """Groups states by (l_id, node) into a dictionary of lists with structure {l_id: {node: [states]}}."""
        #TODO: change rezStates_minus_by_graph to rezStates_minus_by_node
        self.rezStates_minus_by_node = defaultdict(lambda: defaultdict(list))  # Nested defaultdict for automatic list initialization

        # Group states by l_id and node
        # TODO: change self.res_states to self.rez_states_minus
        for state in self.rez_states_minus:
            self.rezStates_minus_by_node[state.l_id][state.node].append(state)

        # Check that each l_id has exactly one source and one sink
        for l_id in self.rezStates_minus_by_node:
            source_count = len(self.rezStates_minus_by_node[l_id].get(-1, []))
            sink_count = len(self.rezStates_minus_by_node[l_id].get(-2, []))

            if source_count != 1 or sink_count != 1:
                raise ValueError(f"Graph {l_id} must have exactly one source and one sink, but found {source_count} source(s) and {sink_count} sink(s).")

    def init_defualt_jy_options(self):

        self.jy_options=dict()
        self.jy_options['epsilon']=.00001
        self.jy_options['tolerance_compress']=.00001
        



    #def OLD_init(self,my_PGM_graph_list,prob_RHS,rezStates_minus_by_node,rez_actions_minus,incumbant_lp,jy_options):

       # self.my_PGM_graph_list=my_PGM_graph_list #list of all of the PGM graphs
       # self.prob_RHS=prob_RHS #RHS
       # self.rezStates_minus_by_node=rezStates_minus_by_node #has all of the states in rez states minus by graph . So if I put in a grpah id then i get out the rez states minus to initialize
       # self.rez_actions_minus=rez_actions_minus #get all actions that are currently under consdieration
       # self.incumbant_lp=incumbant_lp# has incumbent LP objective
       # self.jy_options=jy_options #has teh options

    def call_PGM(self):

        #call the PGM
        while(True): #

            [primal_sol,dual_exog,cur_lp]=self.call_PGM_RMP_solver_from_scratch()#we can do better a different time. lets not make it too hard on the first try
            if cur_lp>self.incumbant_lp-self.jy_options['tolerance_compress']: #if we improve the objective then we will do a compression operator
                self.apply_compression_operator(primal_sol) #apply compression
            did_find_neg_red_cost=False #indicate if we found a negative reduced cost 
            for my_graph in self.my_PGM_graph_list: #Iterate over all graphs and find the shortest path
                shortest_path, shortest_path_length, ordered_path_rows=my_graph.construct_specific_pricing_pgm(dual_exog) #construct and call pricing problem
                if shortest_path_length<-self.jy_options['epsilon']: #if we have a negative reduced cost column we will apply expansion
                    self.apply_expansion_operator(shortest_path, shortest_path_length, ordered_path_rows,my_graph)# do teh expasnion operator
                    did_find_neg_red_cost=True #did find negative reduced cost is set to true

            if did_find_neg_red_cost==False: #if we did not find a negative reduced  then we break and we are done 
                break


    def apply_expansion_operator(self, shortest_path, shortest_path_length, ordered_path_rows, my_graph):
        """Expands the solution by adding new states and actions from the shortest path."""

        print(f"Applying expansion operator for graph {my_graph}...")

        # Step 1: Extract state IDs from the shortest path
        path_state_ids = set(shortest_path)  # Get all visited state IDs

        # Step 2: Map state IDs to actual state objects in the graph
        path_states = {
            state_id: self.pgm_graph_2_rmp_graph[my_graph].state_id_to_state[state_id]
            for state_id in path_state_ids
            if state_id in self.pgm_graph_2_rmp_graph[my_graph].state_id_to_state
        }

        # Step 3: Ensure `rezStates_minus_by_node[my_graph]` exists and is structured as a defaultdict(set)
        if my_graph not in self.rezStates_minus_by_node:
            self.rezStates_minus_by_node[my_graph] = defaultdict(set)

        # Step 4: Group states by their associated node
        for state in path_states.values():
            node = state.node  # Assuming each state object has a `node` attribute
            self.rezStates_minus_by_node[my_graph][node].add(state)

        # Step 5: Extract used actions from ordered path rows
        if not hasattr(self, "res_actions"):  # Ensure `res_actions` exists
            self.res_actions = set()

        for _, _, action in ordered_path_rows:
            if action:  # Ensure the action is not None
                self.res_actions.add(action)

        # Step 6: Update `res_states_minus` as the union of all `res_states_minus_by_graph`
        self.res_states_minus = set().union(*self.res_states_minus_by_graph.values())

        print(f"Expansion applied: {len(path_states)} states and {len(self.res_actions)} actions added to graph {my_graph}.")


    def apply_compression_operator(self):
        """Applies the compression operator to filter active variables from the LP solution."""
        
        print('Applying compression operator...')

        # Step 1: Extract all active variables from the LP primal solution
        active_vars = {var_name for var_name, value in self.primal_solution.items() if value > 1e-6}

        # Step 2: Separate actions and edges
        self.rez_actions = set()  # Set of selected actions
        active_edges = set()  # Set of selected edges (g, s1, s2)

        for var_name in active_vars:
            if var_name[0] == "eq_act_var":  # Action variable format: ('eq_act_var', g, eq_class, action)
                _, g, eq_class, action = var_name  # Extract components
                self.rez_actions.add(action)  # Store the action

            elif var_name[0] == "edge":  # Edge variable format: ('edge', g, s1, s2)
                _, g, s1, s2 = var_name  # Extract graph and states
                active_edges.add((g, s1, s2))
        #Remove null action really this is just executed once so hecne the break
        for g in self.my_PGM_graph_list:
            self.rez_actions=self.rez_actions-g.nullAction
            break
        # Step 3: Ensure `rezStates_minus_by_node[g]` exists as a defaultdict(set)
        self.rezStates_minus_by_node = {g: defaultdict(set) for g in self.pgm_graph_2_rmp_graph}  

        # Step 4: Update `rezStates_minus_by_node[g][node]` to include states from active edges
        for g, s1, s2 in active_edges:
            state1 = self.pgm_graph_2_rmp_graph[g].state_id_to_state[s1]  # Retrieve state object
            state2 = self.pgm_graph_2_rmp_graph[g].state_id_to_state[s2]  # Retrieve state object

            if state1.node!=state2.node:
                self.rezStates_minus_by_node[g][state1.node].add(state1)
                self.rezStates_minus_by_node[g][state2.node].add(state2)
        #compute_res_states,
        self.res_states_minus= set().union(*self.res_states_minus_by_graph.values())

    def return_rez_states_minus_and_res_actions(self):
        return self.res_states_minus,self.res_actions

    def call_PGM_RMP_solver_from_scratch(self):
        """Constructs and initializes the RMP solver from scratch."""
        
        # Step 1: Initialize the RMP graphs
        self.pgm_graph_2_rmp_graph = {}  # Dictionary to store RMP graphs
        for g in self.my_PGM_graph_list:
            my_states_g_by_node = self.rezStates_minus_by_node[g]
            self.pgm_graph_2_rmp_graph[g] = RMP_graph_given_l(g, my_states_g_by_node, self.rez_actions_minus, self.null_action_info)
            self.pgm_graph_2_rmp_graph[g].initialize_system()  # Initialize RMP graph

        # Step 2: Initialize variables and constraints
        self.all_vars = []  # List to store all variables
        self.all_con_names = set()  # Set of all constraint names
        self.lbCon = defaultdict(float)  # Lower bounds on constraints
        self.ubCon = defaultdict(float)  # Upper bounds on constraints

        # Step 3: Create exogenous constraints
        for exog_num in range(self.prob_RHS.size):
            exog_name = ('exog', exog_num)
            self.all_con_names.add(exog_name)
            self.lbCon[exog_name] = self.prob_RHS[exog_num]  # Set lower bound
            self.ubCon[exog_name] = np.inf  # Upper bound is infinity

        # Step 4: Create non-exogenous constraints
        for g, rmp_graph in self.pgm_graph_2_rmp_graph.items():
            for my_eq in rmp_graph.equiv_class_2_s1_s2_pairs:
                non_exog_name = ('eq_con', my_eq, g)
                self.ubCon[non_exog_name] = 0
                self.lbCon[non_exog_name] = 0

        # Step 5: Create flow conservation constraints
        for g, rmp_graph in self.pgm_graph_2_rmp_graph.items():
            for my_node in rmp_graph.resStates_by_node:
                for my_state in rmp_graph.resStates_by_node[my_node]:
                    if not my_state.is_source and not my_state.is_sink:
                        non_exog_name = ('flow_con', my_state.state_id, g)
                        self.ubCon[non_exog_name] = 0
                        self.lbCon[non_exog_name] = 0

        # Step 6: Create variables and associated actions
        for g, rmp_graph in self.pgm_graph_2_rmp_graph.items():
            for my_eq in rmp_graph.equiv_class_2_s1_s2_pairs:
                for my_act in rmp_graph.equiv_class_2_actions[my_eq]:
                    my_cost = my_act.cost  # Get cost
                    my_exog = my_act.Exog_vec  # Get exogenous vector
                    my_contrib_dict = defaultdict(float)  # Dictionary for contributions

                    # Use precomputed nonzero indices for efficiency
                    for exog_num in my_act.nonzero_exog_indices:
                        exog_name = ('exog', exog_num)
                        my_contrib_dict[exog_name] = my_act.my_exog[exog_num]

                    # Create constraint for the equivalence class
                    non_exog_name = ('eq_con', my_eq, g)
                    my_contrib_dict[non_exog_name] = -1

                    # Define variable name and store it
                    my_name = ('eq_act_var', g, my_eq, my_act)
                    new_var = jy_var(my_cost, my_exog, my_contrib_dict, my_name)
                    self.all_vars.append(new_var)

                # Step 7: Create variables for state transitions (edges)
                for (s1, s2) in rmp_graph.equiv_class_2_s1_s2_pairs[my_eq]:
                    my_cost = 0
                    my_exog = None  # No exogenous contribution for edges
                    my_contrib_dict = defaultdict(float)

                    non_exog_name = ('eq_con', my_eq, g)
                    my_contrib_dict[non_exog_name] = 1

                    # Flow conservation constraints
                    if not s1.is_source:
                        flow_in_name_exog_name = ('flow_con', s1.state_id, g)
                        my_contrib_dict[flow_in_name_exog_name] = 1
                    if not s2.is_sink:
                        flow_out_name_exog_name = ('flow_con', s2.state_id, g)
                        my_contrib_dict[flow_out_name_exog_name] = -1

                    # Define variable name and store it
                    my_name = ('edge', g, s1, s2)
                    new_var = jy_var(my_cost, my_exog, my_contrib_dict, my_name)
                    self.all_vars.append(new_var)

        return self.construct_and_solve_lp(self.all_vars,self.all_con_names,self.lbCon,self.ubCon)

    def construct_and_solve_lp(self,jy_vars, all_con_names, lbCon, ubCon):
        """Constructs and solves an Xpress Linear Program (LP), returning primal and dual solutions."""

        
        # Step 1: Create an Xpress optimization problem
        prob = xp.problem()
        
        # Step 2: Create Xpress variables
        xpress_vars = {}
        for var in jy_vars:
            xpress_vars[var.my_name] = xp.var(name=var.my_name, lb=0)
        
        # Step 3: Define the Objective Function (Minimize Cost)
        objective = xp.Sum(var.my_cost * xpress_vars[var.my_name] for var in jy_vars)
        prob.objective = objective
        prob.sense = xp.minimize  # Set minimization
        
        # Step 4: Add Constraints
        constraint_dict = {}  # Store constraint objects for dual values
        
        for con_name in all_con_names:
            # Compute constraint sum from contributions
            constraint_expr = xp.Sum(var.my_contrib_dict.get(con_name, 0) * xpress_vars[var.my_name] for var in jy_vars)
            
            # Apply lower and upper bounds if they exist
            if con_name in lbCon:
                constraint_dict[f"LowerBound_{con_name}"] = prob.addconstraint(
                    constraint_expr >= lbCon[con_name], 
                    name=f"LowerBound_{con_name}"
                )
            if con_name in ubCon:
                constraint_dict[f"UpperBound_{con_name}"] = prob.addconstraint(
                    constraint_expr <= ubCon[con_name], 
                    name=f"UpperBound_{con_name}"
                )
        
        # Step 5: Solve the LP
        prob.solve()  # Xpress handles solution parameters internally
        
        # Step 6: Extract Primal Solution (Decision Variable Values)
        primal_solution = {name: var.sol for name, var in xpress_vars.items()}
        
        # Step 7: Extract Dual Solution (Shadow Prices of Constraints)
        dual_solution = {}
        for con_name, constraint in constraint_dict.items():
            dual_solution[con_name] = constraint.dual
        
        # Step 8: Extract Optimal Objective Value
        optimal_value = prob.objective.value
        
        return primal_solution, dual_solution, optimal_value

#given the following i want a function making and LP using Pulp.  I have jy_var defined below.    
# a list of jy_vars called jy_vars (each wiht a lower bound of zero and upper bound of inntinite)
#all_con_names is a set of names of constraints
#lbCon:  maps a constraint name to a lower bound for the contraint RHS
#ubCon:  maps a constraint name to a upper bound for the contraint RHS