from src.common.rmp_graph_given_1 import RMP_graph_given_l
from collections import defaultdict
from src.common.jy_var import jy_var
import numpy as np
import xpress as xp
class PGM_appraoch:

    def __init__(self,my_PGM_graph_list,prob_RHS,rezStates_minus_by_graph,rez_actions_minus,incumbant_lp,jy_options):

        self.my_PGM_graph_list=my_PGM_graph_list
        self.prob_RHS=prob_RHS
        self.rezStates_minus_by_graph=rezStates_minus_by_graph
        self.rez_actions_minus=rez_actions_minus
        self.incumbant_lp=incumbant_lp
        self.jy_options=jy_options

    def call_PGM(self):

        while(True):

            [primal_sol,dual_exog,cur_lp]=self.call_PGM_RMP_solver_from_scratch()#we can do better a different time
            if cur_lp>self.incumbant_lp-self.jy_options.tolerance_compress:
                self.apply_compression_operator(primal_sol)
            did_find_neg_red_cost=False
            for my_graph in self.my_PGM_graph_list:
                shortest_path, shortest_path_length, ordered_path_rows=my_graph.construct_specific_pricing_pgm(dual_exog)
                if shortest_path_length<-self.jy_options.epsilon:
                    self.apply_expansion_operator(shortest_path, shortest_path_length, ordered_path_rows,my_graph)
                    did_find_neg_red_cost=True

            if did_find_neg_red_cost==False:
                break
        return primal_sol,dual_exog,cur_lp

    def apply_expansion_operator(self, shortest_path, shortest_path_length, ordered_path_rows, my_graph):
        """Expands the solution by adding new states from the shortest path to rezStates_minus_by_graph."""
        
        print(f"Applying expansion operator for graph {my_graph}...")

        # Step 1: Extract states from the shortest path
        path_state_ids = set(shortest_path)  # Get all states visited in the path

        # Step 2: Map state IDs to actual state objects in the graph
        path_states = {state_id: state for state_id, state in self.pgm_graph_2_rmp_graph[my_graph].state_id_to_state.items() if state_id in path_state_ids}

        # Step 3: Expand `rezStates_minus_by_graph` by adding missing states
        self.rezStates_minus_by_graph[my_graph].update(path_states.values())

        print(f"Expansion applied: {len(path_states)} states added to rezStates_minus_by_graph[{my_graph}].")

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

        # Step 3: Update `rezStates_minus_by_graph` to include states in at least one active edge
        self.rezStates_minus_by_graph = {g: set() for g in self.pgm_graph_2_rmp_graph}  # Initialize per graph

        for g, s1, s2 in active_edges:
            self.rezStates_minus_by_graph[g].update([s1, s2])

        print(f"Compression applied: {len(self.rez_actions)} actions selected, {len(active_edges)} edges retained.")



    def call_PGM_RMP_solver_from_scratch(self):
        self.pgm_graph_2_rmp_graph=dict()
        for g in self.my_PGM_graph_list.items():
            my_states_g=self.rezStates_minus_by_graph[g]
            self.pgm_graph_2_rmp_graph[g]=RMP_graph_given_l(g,my_states_g,self.rez_actions_minus)
            self.pgm_graph_2_rmp_graph[g].initialize_system()
            
        self.all_vars=[]
        self.all_con_names=set()
        self.lbCon=defaultdict()
        self.ubCon=defaultdict()


        for exog_num in range(0,self.prob_RHS.size):
            exog_name=tuple(['exog',exog_num])
            self.lbCon[exog_name]=-np.inf
            self.ubCon[exog_name]=self.prob_RHS[exog_num]
        for g in self.pgm_graph_2_rmp_graph.items():
            for my_eq in g.equiv_class_2_s1_s2_pairs:
                for my_act in g.equiv_class_2_actions[my_eq]:
                    my_cost=my_act.cost
                    my_exog=my_act.Exog_vec
                    my_contrib_dict=defaultdict(0)

                    for exog_num in range(0,my_exog.size):
                        exog_name=tuple(['exog',exog_num])
                        self.all_con_names.add(exog_name)
                        my_contrib_dict[exog_name]=my_exog[exog_num]
                    non_exog_name=tuple(['eq_con',my_eq,g])
                    my_contrib_dict[non_exog_name]=-1
                    self.all_con_names.add(non_exog_name)
                    my_name=tuple(['eq_act_var',g,my_eq,my_act])
                    new_var=jy_var(my_cost,my_exog,my_contrib_dict,my_name)
                    self.all_vars.append(new_var)
                for (s1,s2) in g.equiv_class_2_s1_s2_pairs[my_eq]:
                    my_cost=0
                    my_exog=0
                    my_contrib_dict=defaultdict(0)
                    non_exog_name=tuple(['eq_con',my_eq,g])
                    my_contrib_dict[non_exog_name]=1
                    if s1.is_source==False:
                        flow_in_name_exog_name=tuple(['flow_con',s1.state_id,g])
                        my_contrib_dict[flow_in_name_exog_name]=1
                    if s2.is_sink==False:
                        flow_out_name_exog_name=tuple(['flow_con',s2.state_id,g])
                        my_contrib_dict[flow_out_name_exog_name]=-1
                    my_name=tuple(['edge',g,s1,s2])
                    new_var=jy_var(my_cost,my_exog,my_contrib_dict,my_name)
                    self.all_vars.append(new_var)


    def construct_and_solve_lp(jy_vars, all_con_names, lbCon, ubCon):
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
        prob.sense = xp.minimize
        
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
        prob.solve()
        
        # Step 6: Extract Primal Solution (Decision Variable Values)
        primal_solution = {name: var.sol for name, var in xpress_vars.items()}
        
        # Step 7: Extract Dual Solution (Shadow Prices of Constraints)
        dual_solution = {}
        for con_name, constraint in constraint_dict.items():
            dual_solution[con_name] = constraint.dual
        
        # Step 8: Extract Optimal Objective Value
        optimal_value = prob.objective.value
        
        return primal_solution, dual_solution, optimal_value