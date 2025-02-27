from src.common.rmp_graph_given_1 import RMP_graph_given_l
from src.common.full_multi_graph_object_given_l import Full_Multi_Graph_Object_given_l
from src.common.action import Action
from src.common.state import State
from collections import defaultdict
from src.common.jy_var import jy_var
from typing import Dict, DefaultDict, Set, List
import numpy as np
import pulp as pl

import xpress as xp

class PGM_appraoch:
    #This will do the enitre RMP. 

    #This will do the enitre RMP. 

    def __init__(self,index_to_graph:dict[Full_Multi_Graph_Object_given_l],prob_RHS,rez_states_minus: Set[State],rez_actions_minus,incumbant_lp,dominated_action,null_action_info):
        self.index_to_graph:DefaultDict[int,Full_Multi_Graph_Object_given_l] = index_to_graph
        self.my_PGM_graph_list:List[Full_Multi_Graph_Object_given_l]=list(self.index_to_graph.values()) #list of all of the PGM graphs
        self.prob_RHS:np.ndarray=prob_RHS #RHS

        self.rez_states_minus:Set[State]=rez_states_minus #has all of the states in rez states minus by graph . So if I put in a grpah id then i get out the rez states minus to initialize
        self.make_res_states_minus_by_node()
        self.rez_actions_minus=rez_actions_minus #get all actions that are currently under consdieration
        self.incumbant_lp=incumbant_lp# has incumbent LP objective
        self.null_action_info = null_action_info
        self.dominated_actions = dominated_action
        self.init_defualt_jy_options()
        self.primal_solution, self.dual_solution, self.optimal_value = None, None, None

        
    def make_res_states_minus_by_node(self):
        """Groups states by (l_id, node) into a dictionary of lists with structure {l_id: {node: [states]}}."""

        self.rezStates_minus_by_node = defaultdict(lambda: defaultdict(list))  # Nested defaultdict for automatic list initialization
        self.res_states_minus_by_graph: Dict[int, Set[State]] = defaultdict(set)
        for state in self.rez_states_minus:
            self.rezStates_minus_by_node[state.l_id][state.node].append(state)
            self.res_states_minus_by_graph[state.l_id].add(state)
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

            [self.primal_sol,self.dual_exog,self.cur_lp]=self.call_PGM_RMP_solver_from_scratch()#we can do better a different time. lets not make it too hard on the first try
            if self.cur_lp>self.incumbant_lp-self.jy_options['tolerance_compress']: #if we improve the objective then we will do a compression operator
                self.apply_compression_operator() #apply compression
            did_find_neg_red_cost=False #indicate if we found a negative reduced cost 
            for my_graph in self.my_PGM_graph_list: #Iterate over all graphs and find the shortest path
                shortest_path, shortest_path_length, ordered_path_rows=my_graph.construct_specific_pricing_pgm(self.dual_exog,self.rezStates_minus_by_node) #construct and call pricing problem
                print('shortest_path')
                print(shortest_path)
                print('shortest_path_length')
                print(shortest_path_length)
                if shortest_path_length<-self.jy_options['epsilon']: #if we have a negative reduced cost column we will apply expansion
                    self.apply_expansion_operator(shortest_path, shortest_path_length, ordered_path_rows,my_graph)# do teh expasnion operator
                    did_find_neg_red_cost=True #did find negative reduced cost is set to true

            if did_find_neg_red_cost==False: #if we did not find a negative reduced  then we break and we are done 
                break


    def apply_expansion_operator(self, shortest_path, shortest_path_length, ordered_path_rows, my_graph: Full_Multi_Graph_Object_given_l):
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
        if my_graph.l_id not in self.rezStates_minus_by_node:
            self.rezStates_minus_by_node[my_graph.l_id] = defaultdict(set)

        # Step 4: Group states by their associated node
        for state in path_states.values():
            node = state.node  # Assuming each state object has a `node` attribute
            self.rezStates_minus_by_node[my_graph.l_id][node].add(state)

        # Step 5: Extract used actions from ordered path rows
        #if not hasattr(self, "rez_actions"):  # Ensure `res_actions` exists
        #    self.rez_actions = set()

        for _, _, action in ordered_path_rows:
            if action:  # Ensure the action is not None
                self.rez_actions_minus.add(action)

        # Step 6: Update `res_states_minus` as the union of all `res_states_minus_by_graph`
        self.res_states_minus = set().union(*self.res_states_minus_by_graph.values())



    def apply_compression_operator(self):
        """Applies the compression operator to filter active variables from the LP solution."""
        
        print('Applying compression operator...')

        # Step 1: Extract all active variables from the LP primal solution
        active_vars = {var_name for var_name, value in self.primal_sol.items() if value > 1e-6}

        # Step 2: Separate actions and edges
        self.rez_actions_minus = set()  # Set of selected actions
        active_edges = set()  # Set of selected edges (g, s1, s2)

        for var_name in active_vars:
            if var_name[0] == "eq_act_var":  # Action variable format: ('eq_act_var', g, eq_class, action)
                _, g, eq_class, action = var_name  # Extract components
                self.rez_actions_minus.add(action)  # Store the action

            elif var_name[0] == "edge":  # Edge variable format: ('edge', g, s1, s2)
                _, g, s1, s2 = var_name  # Extract graph and states
                active_edges.add((g, s1, s2))
        #Remove null action really this is just executed once so hecne the break
        for g in self.my_PGM_graph_list:
            self.rez_actions_minus=self.rez_actions_minus-g.null_action
            break
        # Step 3: Ensure `rezStates_minus_by_node[g]` exists as a defaultdict(set)
        self.rezStates_minus_by_node = {g: defaultdict(set) for g in self.pgm_graph_2_rmp_graph}  

        # Step 4: Update `rezStates_minus_by_node[g][node]` to include states from active edges
        for g, s1_id, s2_id in active_edges:
            state1 = self.pgm_graph_2_rmp_graph[g].state_id_to_state[s1_id]  # Retrieve state object
            state2 = self.pgm_graph_2_rmp_graph[g].state_id_to_state[s2_id]  # Retrieve state object

            if state1.node!=state2.node:
                self.rezStates_minus_by_node[g][state1.node].add(state1)
                self.rezStates_minus_by_node[g][state2.node].add(state2)
        #compute_res_states,
        self.res_states_minus= set().union(*self.res_states_minus_by_graph.values())

    def return_rez_states_minus_and_res_actions(self):
        return self.rez_states_minus,self.rez_actions_minus

    def call_PGM_RMP_solver_from_scratch(self):
        """Constructs and initializes the RMP solver from scratch."""
        
        # Step 1: Initialize the RMP graphs
        self.pgm_graph_2_rmp_graph:DefaultDict[Full_Multi_Graph_Object_given_l,RMP_graph_given_l] = defaultdict()  # Dictionary to store RMP graphs

        for l_id, g in self.index_to_graph.items():
            
            my_states_g_by_node = self.rezStates_minus_by_node[l_id]
            self.pgm_graph_2_rmp_graph[g] = RMP_graph_given_l(g, my_states_g_by_node, self.rez_actions_minus, self.dominated_actions,self.null_action_info)
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
            #self.ubCon[exog_name] = np.inf  # Upper bound is infinity

        # Step 4: Create non-exogenous constraints
        for g, rmp_graph in self.pgm_graph_2_rmp_graph.items():
            for my_eq in rmp_graph.equiv_class_2_s1_s2_pairs:
                non_exog_name = ('eq_con', my_eq, g.l_id)
                self.all_con_names.add(non_exog_name)

                self.ubCon[non_exog_name] = 0
                self.lbCon[non_exog_name] = 0

        # Step 5: Create flow conservation constraints
        for g, rmp_graph in self.pgm_graph_2_rmp_graph.items():
            for my_node in rmp_graph.resStates_minus_by_node:
                for my_state in rmp_graph.resStates_minus_by_node[my_node]:
                    if not my_state.is_source and not my_state.is_sink:
                        non_exog_name = ('flow_con', my_state.state_id, g.l_id)
                        self.all_con_names.add(non_exog_name)

                        self.ubCon[non_exog_name] = 0
                        self.lbCon[non_exog_name] = 0
        # Step 6: Create variables and associated actions
        for g, rmp_graph in self.pgm_graph_2_rmp_graph.items():
            #input('1  i should make it here lots of times')

            for my_eq in rmp_graph.equiv_class_2_s1_s2_pairs:
                #input('2 i should make it here lots of times')

                for my_act in rmp_graph.equiv_class_2_actions[my_eq]:
                    #input('3  i should make it here lots of times')
                    my_cost = my_act.cost  # Get cost
                    my_exog = my_act.Exog_vec  # Get exogenous vector
                    my_contrib_dict = defaultdict()  # Dictionary for contributions

                    # Use precomputed nonzero indices for efficiency
                    for exog_num in my_act.non_zero_indices_exog:
                        exog_name = ('exog', exog_num)
                        my_contrib_dict[exog_name] = my_act.Exog_vec[exog_num]

                    # Create constraint for the equivalence class
                    non_exog_name = ('eq_con', my_eq, g.l_id)
                    my_contrib_dict[non_exog_name] = -1
                    
                    # Define variable name and store it
                    my_name = ('eq_act_var', g.l_id, my_eq, my_act.action_id)
                    #TODO: remove my_exog here
                    #new_var = jy_var(my_cost, my_exog, my_contrib_dict, my_name)
                    new_var = jy_var(my_cost, my_contrib_dict, my_name)
                    self.all_vars.append(new_var)

                # Step 7: Create variables for state transitions (edges)
                for (s1, s2) in rmp_graph.equiv_class_2_s1_s2_pairs[my_eq]:
                    my_cost = 0
                    my_exog = None  # No exogenous contribution for edges
                    my_contrib_dict = defaultdict(float)

                    non_exog_name = ('eq_con', my_eq, g.l_id)
                    my_contrib_dict[non_exog_name] = 1
                    
                    # Flow conservation constraints
                    
                    if not s1.is_source:
                        flow_in_name_exog_name = ('flow_con', s1.state_id, g.l_id)
                        my_contrib_dict[flow_in_name_exog_name] = 1
                        
                    if not s2.is_sink:
                        flow_out_name_exog_name = ('flow_con', s2.state_id, g.l_id)
                        my_contrib_dict[flow_out_name_exog_name] = -1
                        

                    # Define variable name and store it
                    my_name = ('edge', g.l_id, s1.state_id, s2.state_id)
                    #TODO: remove my_exog here
                    new_var = jy_var(my_cost, my_contrib_dict, my_name)
                    self.all_vars.append(new_var)
                    #print('new_var')
                    #print(new_var)
                    #input('----')
        


        primal_solution, dual_solution, optimal_value = self.solve_with_pulp(self.all_vars,self.all_con_names,self.lbCon,self.ubCon)
        
        dual_exog=np.zeros(self.prob_RHS.size)
        for exog_num in range(self.prob_RHS.size):
            exog_name = ('exog', exog_num)
            exog_name_aug="LowerBound_"+str(exog_name)
            #print('exog_name')
            #print(exog_name)
            #print('exog_name_aug')
            #print(exog_name_aug)
            #print('exog_name_aug in dual_solution')
            #print(exog_name_aug in dual_solution)
            dual_exog[exog_num]=dual_solution[exog_name_aug]
        
        return primal_solution, dual_exog, optimal_value

    def solve_with_pulp(self, jy_vars, all_con_names, lbCon, ubCon):
        # Step 1: Create a PuLP minimization problem
        prob = pl.LpProblem(name="OptimizationProblem", sense=pl.LpMinimize)
        
        # Step 2: Create PuLP variables
        pulp_vars = {}
        for var in jy_vars:
            var_name = var.my_name
            pulp_vars[var_name] = pl.LpVariable(name=str(var_name), lowBound=0, cat='Continuous')
        
        # Step 3: Define the Objective Function (Minimize Cost)
        objective = pl.lpSum(var.my_cost * pulp_vars[var.my_name] for var in jy_vars)
        prob += objective
        
        # Step 4: Add Constraints
        constraint_dict = {}  # Store constraint objects for dual values
        constraint_mapping = {}  # Map from original constraint name to the actual PuLP constraint name
        
        for con_name in all_con_names:
            # Compute constraint sum from contributions
            constraint_expr = pl.lpSum(var.my_contrib_dict.get(con_name, 0) * pulp_vars[var.my_name] for var in jy_vars)
            
            # Apply lower and upper bounds if they exist
            if con_name in lbCon:
                constraint = constraint_expr >= lbCon[con_name]
                # Create a much simpler constraint name using a counter
                constraint_name = f"LB_{len(constraint_dict)}"
                prob += (constraint, constraint_name)
                constraint_dict[constraint_name] = constraint
                constraint_mapping[f"LowerBound_{con_name}"] = constraint_name
                
            if con_name in ubCon:
                constraint = constraint_expr <= ubCon[con_name]
                # Create a much simpler constraint name using a counter
                constraint_name = f"UB_{len(constraint_dict)}"
                prob += (constraint, constraint_name)
                constraint_dict[constraint_name] = constraint
                constraint_mapping[f"UpperBound_{con_name}"] = constraint_name
        
        # Step 4.5: Print the model formulation
        self.print_pulp_formulation(prob)
        
        # Step 5: Solve the problem
        prob.solve()
        
        # Step 6: Extract solutions
        primal_solution = {}
        for var_name, pulp_var in pulp_vars.items():
            primal_solution[var_name] = pulp_var.value()
        
        # Extract dual values
        dual_solution = {}
        
        if prob.status == 1:  # If the problem was solved optimally
            # Print all constraint names that PuLP knows about
            print("Available constraints in PuLP:", list(prob.constraints.keys()))
            
            # First, get all the duals using the simplified names we created
            temp_duals = {}
            for simplified_name in constraint_dict.keys():
                try:
                    if simplified_name in prob.constraints:
                        temp_duals[simplified_name] = prob.constraints[simplified_name].pi
                    else:
                        # Try a direct lookup in constraints dictionary
                        found = False
                        for name in prob.constraints:
                            if simplified_name in name:  # Check if our simplified name is part of the actual name
                                temp_duals[simplified_name] = prob.constraints[name].pi
                                found = True
                                print(f"Found constraint {simplified_name} as {name}")
                                break
                        
                        if not found:
                            print(f"Warning: Could not find constraint: {simplified_name}")
                            temp_duals[simplified_name] = 0
                except Exception as e:
                    print(f"Error getting dual for {simplified_name}: {e}")
                    temp_duals[simplified_name] = 0
            
            # Now map back to the original constraint names format
            for original_name, simplified_name in constraint_mapping.items():
                dual_solution[original_name] = temp_duals.get(simplified_name, 0)
        else:
            # If the problem wasn't solved optimally, set all duals to 0
            for original_name in constraint_mapping.keys():
                dual_solution[original_name] = 0
        
        dual_sol = np.array(list(dual_solution.values()))
        
        # Get optimal objective value
        optimal_value = pl.value(prob.objective)

        
        return primal_solution, dual_solution, optimal_value

    def construct_and_solve_lp(self, jy_vars, all_con_names, lbCon, ubCon):
        """
        Constructs and solves an Xpress Linear Program (LP), returning primal and dual solutions.
        """
        import xpress as xp
        # Step 1: Initialize Xpress problem
        xp.init('C:/xpressmp/bin/xpauth.xpr')  # Ensure Xpress is initialized correctly
        prob = xp.problem()

        # Step 2: Create Xpress variables
        xpress_vars = {}

        for var in jy_vars:
            var_name = var.my_name  # Ensure variable name is a string
            xpress_vars[var_name] = xp.var(vartype=xp.continuous, name=str(var_name),lb=0,ub=float("inf"))  # Create variable
        prob.addVariable(list(xpress_vars.values()))  # Add variables to the problem

        # Step 3: Define the Objective Function (Minimize Cost)
        objective = xp.Sum(var.my_cost * xpress_vars[var.my_name] for var in jy_vars)
        prob.setObjective(objective, sense=xp.minimize)

        # Step 4: Add Constraints
        constraint_dict = {}  # Store constraint objects for dual values
        for con_name in all_con_names:
            # Compute constraint sum from contributions
            constraint_expr = xp.Sum(var.my_contrib_dict.get(con_name, 0) * xpress_vars[var.my_name] for var in jy_vars)

            # Apply lower and upper bounds if they exist
            if con_name in lbCon:
                constraint = xp.constraint(constraint_expr >= lbCon[con_name], name=f"LowerBound_{con_name}")
                prob.addConstraint(constraint)
                constraint_dict[f"LowerBound_{con_name}"] = constraint

            if con_name in ubCon:
                constraint = xp.constraint(constraint_expr <= ubCon[con_name], name=f"UpperBound_{con_name}")
                prob.addConstraint(constraint)
                constraint_dict[f"UpperBound_{con_name}"] = constraint

        


        prob.solve()
        
        primal_solution = {}

        # Iterate through the xpress_vars dictionary
        for var_name, xp_var in xpress_vars.items():
            # Get the solution value for each variable
            primal_solution[var_name] = prob.getSolution(xp_var)


        dual_solution = {}

        # Iterate through the constraint_dict dictionary
        for con_name, constraint in constraint_dict.items():
            # Get the dual value for each constraint
            dual_solution[con_name] = prob.getDual(constraint)

        dual_sol = np.array(list(dual_solution.values()))

        optimal_value = prob.getObjVal()



        return primal_solution, dual_sol, optimal_value

    def solve_with_pulp_not_use(self,jy_vars, all_con_names, lbCon, ubCon):
        # Step 1: Create a PuLP minimization problem
        prob = pl.LpProblem(name="OptimizationProblem", sense=pl.LpMinimize)
        
        # Step 2: Create PuLP variables
        pulp_vars = {}
        for var in jy_vars:
            var_name = var.my_name
            pulp_vars[var_name] = pl.LpVariable(name=str(var_name), lowBound=0, cat='Continuous')
        
        # Step 3: Define the Objective Function (Minimize Cost)
        objective = pl.lpSum(var.my_cost * pulp_vars[var.my_name] for var in jy_vars)
        prob += objective
        
        # Step 4: Add Constraints
        constraint_dict = {}  # Store constraint objects for dual values
        constraint_mapping = {}  # Map from constraint name to the actual PuLP constraint name
        
        for con_name in all_con_names:
            # Compute constraint sum from contributions
            constraint_expr = pl.lpSum(var.my_contrib_dict.get(con_name, 0) * pulp_vars[var.my_name] for var in jy_vars)
            
            # Apply lower and upper bounds if they exist
            if con_name in lbCon:
                constraint = constraint_expr >= lbCon[con_name]
                # Use a simple string for the constraint name to avoid tuple formatting issues
                constraint_name = f"LB_{str(con_name).replace('(', '').replace(')', '').replace(',', '_').replace(' ', '')}"
                prob += (constraint, constraint_name)
                constraint_dict[constraint_name] = constraint
                constraint_mapping[f"LowerBound_{con_name}"] = constraint_name
                
            if con_name in ubCon:
                constraint = constraint_expr <= ubCon[con_name]
                # Use a simple string for the constraint name to avoid tuple formatting issues
                constraint_name = f"UB_{str(con_name).replace('(', '').replace(')', '').replace(',', '_').replace(' ', '')}"
                prob += (constraint, constraint_name)
                constraint_dict[constraint_name] = constraint
                constraint_mapping[f"UpperBound_{con_name}"] = constraint_name
        
        # Step 5: Solve the problem
        prob.solve()
        
        # Step 6: Extract solutions
        primal_solution = {}
        for var_name, pulp_var in pulp_vars.items():
            primal_solution[var_name] = pulp_var.value()
        
        # Extract dual values
        dual_solution = {}
        
        if prob.status == 1:  # If the problem was solved optimally
            # First, get all the duals using the simplified names we created
            temp_duals = {}
            for simplified_name in constraint_dict.keys():
                try:
                    if simplified_name in prob.constraints:
                        temp_duals[simplified_name] = prob.constraints[simplified_name].pi
                    else:
                        print(f"Warning: Could not find constraint: {simplified_name}")
                        temp_duals[simplified_name] = 0
                except Exception as e:
                    print(f"Error getting dual for {simplified_name}: {e}")
                    temp_duals[simplified_name] = 0
            
            # Now map back to the original constraint names format
            for original_name, simplified_name in constraint_mapping.items():
                dual_solution[original_name] = temp_duals.get(simplified_name, 0)
        else:
            # If the problem wasn't solved optimally, set all duals to 0
            for original_name in constraint_mapping.keys():
                dual_solution[original_name] = 0
        
        #dual_sol = np.array(list(dual_solution.values()))
        
        # Get optimal objective value
        optimal_value = pl.value(prob.objective)
        
        #self.print_pulp_formulation(prob)

        return primal_solution, dual_solution, optimal_value

 
    def print_pulp_formulation(self,prob):

        """

        Print the formulation of a PuLP model with variable names, coefficients, and RHS values.

        Args:

            prob: A PuLP problem object

        """

        print("=" * 80)

        print("MODEL FORMULATION")

        print("=" * 80)

        # Print objective function

        print("\nOBJECTIVE FUNCTION:")

        if prob.sense == 1:  # Minimize

            print("Minimize:")

        else:

            print("Maximize:")

        obj_terms = []

        for var, coeff in prob.objective.items():

            if coeff != 0:  # Only include non-zero coefficients

                if abs(coeff) == 1:

                    sign = "-" if coeff < 0 else "+"

                    obj_terms.append(f"{sign} {var.name}")

                else:

                    sign = "-" if coeff < 0 else "+"

                    obj_terms.append(f"{sign} {abs(coeff)} {var.name}")

        # Format the objective function nicely

        obj_str = " ".join(obj_terms)

        # Replace leading "+" with empty string if it exists

        obj_str = obj_str[2:] if obj_str.startswith("+ ") else obj_str

        print(f"    {obj_str}")

        # Print constraints

        print("\nCONSTRAINTS:")

        for name, constraint in prob.constraints.items():

            con_terms = []

            # Get the left-hand side terms

            for var, coeff in constraint.items():

                if coeff != 0:  # Only include non-zero coefficients

                    if abs(coeff) == 1:

                        sign = "-" if coeff < 0 else "+"

                        con_terms.append(f"{sign} {var.name}")

                    else:

                        sign = "-" if coeff < 0 else "+"

                        con_terms.append(f"{sign} {abs(coeff)} {var.name}")

            # Format the constraint left-hand side

            con_str = " ".join(con_terms)

            # Replace leading "+" with empty string if it exists

            con_str = con_str[2:] if con_str.startswith("+ ") else con_str

            # Determine constraint sense and right-hand side

            sense = ""

            rhs = 0

            if constraint.sense == 1:  # ≤

                sense = "<="

                rhs = constraint.constant * -1

            elif constraint.sense == -1:  # ≥

                sense = ">="

                rhs = constraint.constant * -1

            else:  # ==

                sense = "="

                rhs = constraint.constant * -1

            print(f"[{name}]: {con_str} {sense} {rhs}")

        # Print variable bounds

        print("\nVARIABLE BOUNDS:")

        for var in prob.variables():

            lb = var.lowBound if var.lowBound is not None else "-inf"

            ub = var.upBound if var.upBound is not None else "+inf"

            if lb == ub:

                print(f"{var.name} = {lb}")

            else:

                print(f"{lb} <= {var.name} <= {ub}")

        # Print variable types

        print("\nVARIABLE TYPES:")

        for var in prob.variables():

            var_type = "Continuous"

            if var.cat == pl.LpInteger:

                var_type = "Integer"

            elif var.cat == pl.LpBinary:

                var_type = "Binary"

            print(f"{var.name}: {var_type}")

        print("=" * 80)
    
