import numpy as np
from collections import defaultdict
from src.common.pgm_approach import Route
import itertools
from src.common.state import State
from scipy.sparse import coo_matrix
import xpress as xp
import os
import platform
from sortedcontainers import SortedList

class xy_jy_cg_solver:
    def __init__(self, list_of_route:list[Route], node_sequence_of_routes, rhs_exog_vec, data, forbidden=[], initial_resource_vector=None):
        self.list_of_route = list_of_route
        self.list_of_action_list = []
        self.node_sequence_of_routes = node_sequence_of_routes
        self.rhs_exog_vec = rhs_exog_vec
        self.var_to_obj_coef = defaultdict()
        self.var_to_col_coef = defaultdict()
        self.variables = {}
        self.col_of_path = 0
        self.col_of_omega = 0
        self.omega_name_to_index = defaultdict()
        self.index_to_omega_name = defaultdict()
        self.index_to_route_name = defaultdict()
        self.route_name_to_index = defaultdict()
        self.row_to_col_to_data = defaultdict(lambda:defaultdict(float))
        self.actions = data.actions
        self.max_action_cost = data.max_action_cost
        self.pickup_node = data.pickup_node
        self.dropoff_node = data.dropoff_node
        self.neighbors = data.neighbors
        self.forbidden = set(forbidden)
        self.initial_resource_vector = initial_resource_vector
        self.rho = self._generate_rho()
        self._initiate_col_coeff()

        self.time_profile = defaultdict(float)
        if platform.system() == 'Windows':
            self.license_path = 'C:/xpressmp/bin/xpauth.xpr'
        else:
            self.license_path = '/mnt/c/xpressmp/bin/xpauth.xpr'

    def _generate_rho(self):
        this_rho = defaultdict()
        for u in self.pickup_node:
            for v in self.pickup_node:
                if u!= v and (u,v) in self.actions and (u+len(self.pickup_node),v+len(self.pickup_node)) in self.actions:
                    this_rho[(u,v)] = 2*self.actions[(u,v)][0].cost + 2*self.actions[(u+len(self.pickup_node),v+len(self.pickup_node))][0].cost
                    this_rho[(u,v)]=(this_rho[(u,v)]*1.01)+1
                else:
                    this_rho[(u,v)] = self.max_action_cost*2
        return this_rho
        
    def _initiate_col_coeff(self):
        var_index = 0
        data =[]
        cols=[]
        rows=[]
        for pi in range(len(self.list_of_route)):
            route = self.list_of_route[pi]
            self.var_to_obj_coef[var_index] = route.cost
            this_exog_vec = route.Exog_vec
            non_zero_indices = np.nonzero(this_exog_vec)[0]
            
            for row in non_zero_indices:
                rows.append(row)
                cols.append(var_index)
                data.append(this_exog_vec[row])
                self.row_to_col_to_data[row][var_index]  = this_exog_vec[row]
            self.index_to_route_name[var_index] = (f'route',pi)
            self.route_name_to_index[(f'route',pi)] = var_index
            var_index += 1
            self.col_of_path += 1
        for u in self.pickup_node:
            for v in set(self.neighbors[u]) & set(self.pickup_node):
                if (u,v) not in self.forbidden:
                    self.var_to_obj_coef[var_index] = self.rho[(u,v)]
                    rows.append(u-1)
                    cols.append(var_index)
                    data.append(-1)
                    self.row_to_col_to_data[u-1][var_index] = -1
                    rows.append(v-1)
                    cols.append(var_index)
                    data.append(1)
                    self.row_to_col_to_data[v-1][var_index] = 1
                    self.omega_name_to_index[('omega',u,v)] = var_index
                    self.index_to_omega_name[var_index] = ('omega',u,v)
                    var_index += 1
                    self.col_of_omega +=1
        self.index_sorted_slot = SortedList()
        self.constraints_matrix = coo_matrix((data, (rows, cols)), shape=(len(self.rhs_exog_vec), var_index))

    def _construct_problem(self):
        """
        Construct the master problem with the generated columns using Xpress.
        Optimized for performance with large-scale problems.
        """
        # Only initialize Xpress once if needed
        try:
            # Initialize the model
            if not hasattr(self, 'xpress_initialized'):
                
                xp.init(self.license_path)
                self.xpress_initialized = True
            
            # Create a new problem
            self.model = xp.problem()
            self.model.setControl("THREADS",1)
            self.model.setControl("DEFAULTALG",4)
            # Pre-allocate variable arrays for batch creation
            route_indices = list(self.index_to_route_name.keys())
            omega_indices = list(self.index_to_omega_name.keys())
            all_indices = route_indices + omega_indices
            
            # Create all variables in batch where possible
            self.variables = {}
            # debug here
            # overlap = set(route_indices) & set(omega_indices)
            # if overlap:
            #     raise ValueError(f"Overlapping indices found: {overlap}")
            # Add route variables
            if route_indices:
                route_vars = [xp.var(name=f"r_{idx}", lb=0) for idx in route_indices]
                for idx, var in zip(route_indices, route_vars):
                    self.variables[idx] = var
                self.model.addVariable(route_vars)  # Add variables in batch if supported
                
            # Add omega variables
            if omega_indices:
                omega_vars = [xp.var(name=f"w_{idx}", lb=0) for idx in omega_indices]
                for idx, var in zip(omega_indices, omega_vars):
                    self.variables[idx] = var
                self.model.addVariable(omega_vars)  # Add variables in batch if supported
            
            # Build the objective expression more efficiently
            obj_terms = []
            for var_idx in all_indices:
                if var_idx in self.var_to_obj_coef:
                    coef = float(self.var_to_obj_coef[var_idx])
                    obj_terms.append(coef * self.variables[var_idx])
            
            obj_expr = sum(obj_terms)
            self.model.setObjective(obj_expr, sense=xp.minimize)
            
            # Build constraints more efficiently
            self.constraints = []
            constraint_exprs = []
            
            # Find non-empty rows to avoid creating empty constraints
            non_empty_rows = [row for row in range(len(self.rhs_exog_vec)) 
                            if row in self.row_to_col_to_data and self.row_to_col_to_data[row]]
            
            for row in non_empty_rows:
                col_to_data = self.row_to_col_to_data[row]
                
                # Collect terms
                terms = []
                for var_index, coef in col_to_data.items():
                    if var_index in self.variables:  # Make sure the variable exists
                        terms.append(coef * self.variables[var_index])
                
                # Create constraint expression
                if terms:
                    constraint_expr = sum(terms) >= self.rhs_exog_vec[row]
                    constraint_exprs.append(constraint_expr)
                    self.constraints.append(constraint_expr)
            
            # Add all constraints at once if possible
            if constraint_exprs:
                self.model.addConstraint(constraint_exprs)
            
            print(f"Problem constructed with {len(self.variables)} variables and {len(self.constraints)} constraints")
            return True
            
        except Exception as e:
            print(f"Error constructing problem: {e}")
            import traceback
            traceback.print_exc()
            return False

    def solve(self):
        """Solve the current master problem using Xpress."""
        self._construct_problem()

        if not hasattr(self, 'model'):
            raise ValueError("Master problem not initialized.")
        
        # Set solver parameters
        try:
            # Use only the most basic control settings
            self.model.setControl({'outputlog': 1})  # Enable output logging
        except Exception as e:
            print(f"Warning: Error setting control parameters: {e}")
            # Proceed without setting controls if they're not supported
        
        # Solve the problem
        #self.model.reset()
        self.model.solve()
        # Get solution status
        status_code = self.model.getProbStatus()
        if status_code == xp.lp_optimal:
            status = "Optimal"
        elif status_code == xp.lp_infeas:
            status = "Infeasible"
        elif status_code == xp.lp_unbound:
            status = "Unbounded"
        else:
            status = f"Other ({status_code})"
        
        # Get objective value
        self.objective_value = self.model.getObjVal() if status == "Optimal" else float('inf')
        
        # Get variable values
        self.var_values = {}
        if status == "Optimal":
            for var_idx, var in self.variables.items():
                var_value = self.model.getSolution(var)
                self.var_values[var_idx] = var_value
        else:
            print(f"Problem status: {status}")
            input('problem not solve')
            
        # Get dual values
        self.dual_values = self.grab_dual_sol()
        
        return {
            'status': status,
            'objective_value': self.objective_value,
            'variable_values': self.var_values,
            'dual_values': self.dual_values
        }
    
    
    def solve_2(self, max_iter=5, max_add=10):
        """
        Solve the RMP problem iteratively according to Algorithm 2.
        
        Parameters:
        max_iter (int): Maximum number of iterations (default: 5)
        max_add (int): Maximum number of columns to add in each iteration
        
        Returns:
        dict: Solution information including status, objective value, variable values
        """
        num_iter_left = max_iter
        col_added = 0
        debug_on = False
        
        while num_iter_left > 0:
            print('check here for loop')
            # Step 3: Solve current RMP
            solution = self.solve()
            
            x_values = solution['variable_values']
            
            # If solution is not optimal, break
            if solution['status'] != 'Optimal':
                break
                
            this_forbidden_omega = self.get_active_DOI()
            if not this_forbidden_omega:
                break
                
            # Step 4: Compute mutation scores for all relevant (l,u,v) triples
            all_mut_scores = []
            my_dual_vals = solution['dual_values']  # Updated to use solution from solve method
            
            # For each route l with x_l > 0
            for var_idx, x_val in x_values.items():
                # Check if this is a route variable
                if var_idx in self.index_to_route_name and x_val > 0.00001:
                    route_name = self.index_to_route_name[var_idx]
                    if route_name[0] == 'route':
                        route_idx = route_name[1]  # Extract the route index from the name tuple
                        route = self.list_of_route[route_idx]
                        
                        if len(route.node_in_ordered) < 4:
                            continue
                        route_nodes = [node for node in route.node_in_ordered if node in self.pickup_node]
                        
                        # For each u in l
                        for u in route_nodes:
                            # For each v not in l where omega_uv > 0
                            for v in self.pickup_node:
                                if v not in route_nodes:
                                    # Check if omega_uv > 0
                                    omega_name = ('omega', u, v)
                                    if omega_name in self.omega_name_to_index:
                                        omega_idx = self.omega_name_to_index[omega_name]
                                        omega_val = x_values.get(omega_idx, 0)
                                        
                                        if omega_val > 0.00001:
                                            # Calculate mutation score
                                            l_hat = self._swap(route, u, v)  # Create new route by swapping u with v
                                            
                                            if l_hat:
                                                this_red_cost = l_hat.get_red_cost(my_dual_vals)
                                                print('this_red_cost')
                                                print(this_red_cost)
                                                
                                                all_mut_scores.append(tuple([this_red_cost, l_hat, route_idx]))
            
            # Step 5: Select columns with negative mutation scores
            cols_to_add = [(score, route, orig_idx) for score, route, orig_idx in all_mut_scores if score <= -.0001]
            
            # Step 6: Remove routes that are already in my_routes
            filtered_cols_to_add = []
            for score, route, orig_idx in cols_to_add:
                if route.node_in_ordered not in self.node_sequence_of_routes:
                    filtered_cols_to_add.append((score, route, orig_idx))
                else:
                    print('Route already exists in node_sequence_of_routes')
            
            cols_to_add = filtered_cols_to_add
            
            # Step 7: Take subset with smallest scores, up to max_add columns
            cols_to_add.sort(key=lambda x: x[0])  # Sort by mutation score (smallest first)
            cols_to_add = cols_to_add[:max_add]
            
            # If no columns to add, break
            if not cols_to_add:
                break
            
            # Step 8: Add selected columns to the problem
            if debug_on == True:
                this_sol = self.solve()
                this_obj_val = this_sol['objective_value']
                before_obj = this_obj_val
                if this_obj_val < .001:
                    print('error here')
                    print('just before additions lp')

            for _, route, _ in cols_to_add:
                # Add route to the problem
                self.add_route(route)  # Assuming this method exists and adds a route correctly
                print('Added route:')
                print('Cost:', route.cost)
                print('Nodes:', route.just_nodes_ordered if hasattr(route, 'just_nodes_ordered') else 'N/A')
                col_added += 1
                
            # Step 9: Decrement iterations counter
            num_iter_left -= 1
            
            if debug_on == True:
                this_sol = self.solve()
                print('this_sol')
                print(this_sol)
                this_obj_val = this_sol['objective_value']
                if this_obj_val > before_obj:
                    print('error here: obj not improving with new col')
                var_index_to_value = this_sol['variable_values']
                
                # Get indices of routes with non-zero values
                path_indices_used = []
                for var_idx, value in var_index_to_value.items():
                    if var_idx in self.index_to_route_name and value > 0.0001:
                        route_name = self.index_to_route_name[var_idx]
                        if route_name[0] == 'route':
                            route_idx = route_name[1]
                            path_indices_used.append(route_idx)
                

                if this_obj_val < .001:
                    print('HOW CAN COST DROP TO ZERO UPON ADDING THESE TERMS')
                print('just after additions lp')

        # Solve one final time with all the added columns
        final_solution = self.solve()
        
        return final_solution, self.node_sequence_of_routes, self.list_of_route
    def solve_ilp(self):
        """
        Solve the integer version of the problem using Xpress, focusing only on route variables.
        Optimized for speed and efficiency.
        """
        try:
            # Initialize Xpress if needed
            if not hasattr(self, 'xpress_initialized'):
                xp.init('C:/xpressmp/bin/xpauth.xpr')
                self.xpress_initialized = True
            
            # Create a new ILP model
            self.ilp_model = xp.problem()
            self.ilp_variables = {}
            
            # Get route variable indices (correct handling of route variables)
            route_indices = [idx for idx, name in self.index_to_route_name.items() 
                            if name[0] == 'route']
            
            if not route_indices:
                print("No route variables found in the model")
                return {'status': 'No routes', 'objective_value': float('inf')}
            
            # Create route variables in batch
            route_vars = []
            for var_idx in route_indices:
                # Create binary variable for each route
                var = xp.var(name=f"r_{var_idx}", lb=0, ub=1, vartype=xp.integer)
                self.ilp_variables[var_idx] = var
                route_vars.append(var)
            
            # Add variables in batch
            self.ilp_model.addVariable(route_vars)
            
            # Build objective expression more efficiently
            obj_terms = []
            for var_idx in route_indices:
                if var_idx in self.var_to_obj_coef:
                    coef = float(self.var_to_obj_coef[var_idx])
                    obj_terms.append(coef * self.ilp_variables[var_idx])
            
            obj_expr = sum(obj_terms)
            self.ilp_model.setObjective(obj_expr, sense=xp.minimize)
            
            # Find non-empty constraints (where at least one route variable appears)
            relevant_rows = set()
            for row in range(len(self.rhs_exog_vec)):
                if row in self.row_to_col_to_data:
                    for var_idx in route_indices:
                        if var_idx in self.row_to_col_to_data[row]:
                            relevant_rows.add(row)
                            break
            
            # Build constraints more efficiently
            self.ilp_constraints = []
            constraint_batch = []
            
            for row in sorted(relevant_rows):
                # Collect terms for this constraint
                terms = []
                for var_idx in route_indices:
                    if var_idx in self.row_to_col_to_data[row]:
                        coef = float(self.row_to_col_to_data[row][var_idx])
                        if abs(coef) > 1e-10:  # Skip tiny coefficients
                            terms.append(coef * self.ilp_variables[var_idx])
                
                # Create constraint if there are terms
                if terms:
                    constraint = sum(terms) >= self.rhs_exog_vec[row]
                    constraint_batch.append(constraint)
                    self.ilp_constraints.append(constraint)
            
            # Add all constraints at once
            if constraint_batch:
                self.ilp_model.addConstraint(constraint_batch)
            
            # Set solver parameters for better performance
            try:
                self.ilp_model.setControl({
                    'presolve': 1,          # Enable presolve
                    'mipgap': 0.005,        # Set 0.5% MIP gap (tighter than before)
                    'timeLimit': 600,       # Set 10-minute time limit
                    'threads': 0,           # Use all available threads
                    'mipEmphasis': 1,       # Emphasize feasibility over optimality
                    'heurFreq': 5,          # Run heuristics more frequently
                    'mipCuts': 2,           # Aggressive cut generation
                    'backtracking': 3,      # Advanced backtracking strategy
                    'outputlog': 1          # Enable output logging
                })
            except Exception as e:
                print(f"Warning: Some control parameters not supported: {e}")
                # Try setting basic controls
                try:
                    self.ilp_model.setControl({'presolve': 1, 'timeLimit': 600})
                except:
                    pass  # Proceed without controls if not supported
            
            print(f"Solving integer program with {len(self.ilp_variables)} variables and {len(self.ilp_constraints)} constraints...")
            
            # Solve the problem
            self.ilp_model.solve()

            
            # Get solution status
            status_code = self.ilp_model.getProbStatus()
            if status_code == xp.mip_optimal:
                status = "Optimal"
            elif status_code == xp.mip_feasible:
                status = "Feasible (not proven optimal)"
            elif status_code == xp.lp_infeas:
                status = "Infeasible"
            else:
                status = f"Other ({status_code})"
            
            # Get objective value
            objective_value = self.ilp_model.getObjVal() if status in ["Optimal", "Feasible (not proven optimal)"] else float('inf')
            
            # Get variable values and used routes more efficiently
            var_values = {}
            route_used = []
            route_indices_used = []
            
            if status in ["Optimal", "Feasible (not proven optimal)"]:
                # Get all solutions at once if possible
                try:
                    all_solutions = self.ilp_model.getSolution(list(self.ilp_variables.values()))
                    for i, var_idx in enumerate(self.ilp_variables.keys()):
                        var_value = all_solutions[i]
                        var_values[var_idx] = var_value
                        
                        # Check if route is used
                        if var_value > 0.5:  # For binary variables
                            route_name = self.index_to_route_name[var_idx]
                            route_idx = route_name[1]  # Extract route index
                            route_indices_used.append(route_idx)
                except:
                    # Fallback to getting solutions one by one
                    for var_idx, var in self.ilp_variables.items():
                        var_value = self.ilp_model.getSolution(var)
                        var_values[var_idx] = var_value
                        
                        # Check if route is used
                        if var_value > 0.5:
                            route_name = self.index_to_route_name[var_idx]
                            route_idx = route_name[1]
                            route_indices_used.append(route_idx)
                
                # Get all used routes at once
                route_used = [self.list_of_route[idx] for idx in route_indices_used]
            else:
                print(f"ILP status: {status} - No solution found")
            
            # Calculate total cost
            total_cost = sum(route.cost for route in route_used)
            
            return {
                'status': status,
                'objective_value': objective_value,
                'model_objective': objective_value,
                'variable_values': var_values,
                'used_routes': route_used,
                'total_cost': total_cost,
                'num_routes': len(route_used),
                'route_indices': route_indices_used  # Added for easier tracking
            }
        
        except Exception as e:
            print(f"Error in solve_ilp: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'Error',
                'objective_value': float('inf'),
                'error_message': str(e)
            }
        
    
    def grab_primal_sol(self):
        return self.objective_value, self.var_values
        
    def grab_dual_sol(self):
        """Extract dual values from the solved model."""
        
        status_code = self.model.getProbStatus()
        if status_code != xp.lp_optimal:
            return None
        
        dual_values = self.model.getDual(self.constraints)

        
        return dual_values
    def get_active_DOI(self):
        """
        Get all active DOI variables (omega variables with non-zero values).
        
        Returns:
        list: List of (u,v) pairs corresponding to active DOI variables
        """
        active_DOI = set()
        for var_index, omega_name in self.index_to_omega_name.items():
            if var_index in self.var_values and abs(self.var_values[var_index]) > 0.00001:
                u = omega_name[1]
                v = omega_name[2]
                active_DOI.add((u,v))
        return active_DOI
    def generate_subset_of_routes(self):
        sub_set_routes = []
        for route in self.list_of_route:
            pickup_nodes_in_route = [node for node in route.node_in_ordered 
                                    if node in self.pickup_node]
            if len(pickup_nodes_in_route) <=2:
                continue
            for size in range(2, len(pickup_nodes_in_route) + 1):
                for pickup_subset in itertools.combinations(pickup_nodes_in_route, size):
                    # Check if this subset forms a valid route
                    if self.is_valid_pickup_subset(pickup_subset, route):
                        # Create a new route
                        new_route = self.create_subset_route(pickup_subset, route)
                        sub_set_routes.append(new_route)
        for route in sub_set_routes:
            state_action_alt_repeat = []
            source_state = State(-1,self.initial_resource_vector,0,True,False)
            state_action_alt_repeat.append(source_state)
            cur_state = source_state
            for (tail,head) in zip(route[:-1],route[1:]):
                this_act = self.actions[(tail,head)][0]
                state_action_alt_repeat.append(this_act)
                next_state = this_act.get_head_state(cur_state)
                if next_state == None:
                    input('error here: none state generated from given column')
                state_action_alt_repeat.append(next_state)
                cur_state = next_state
            this_route = Route(state_action_alt_repeat,1,self.pickup_node)
            self.list_of_route.append(this_route)
    def _swap(self, route, u, v):
        """
        Create a new route by swapping node u with node v.
        
        Parameters:
        route (Route): Original route
        u (int): Node to remove
        v (int): Node to add
        
        Returns:
        Route: New route with u replaced by v, or None if not valid
        """
        # Get the node sequence
        nodes = route.node_in_ordered.copy()
        u_dropoff = u + len(self.pickup_node)
        v_dropoff = v + len(self.pickup_node)
        
        # Find the position of u
        if u not in nodes or u_dropoff not in nodes:
            return None
        
        u_pos = nodes.index(u)
        u_dropoff_pos = nodes.index(u_dropoff)
        
        # Create new node sequence
        new_nodes = nodes.copy()
        new_nodes[u_pos] = v
        new_nodes[u_dropoff_pos] = v_dropoff
        
        if new_nodes in self.node_sequence_of_routes:
            return None
            
        # Create state-action sequence for the new route
        state_action_alt_repeat = []
        source_state = State(-1, self.initial_resource_vector, 0, True, False)
        state_action_alt_repeat.append(source_state)
        cur_state = source_state
        
        for i in range(0, len(new_nodes)-1):
            tail = new_nodes[i]
            head = new_nodes[i+1]
            if (tail, head) not in self.actions:
                return None
            this_act = self.actions[(tail, head)][0]
            state_action_alt_repeat.append(this_act)
            
            next_state = this_act.get_head_state_fast_load_ai(cur_state, cur_state.l_id)
            
            if next_state is None:
                # Not a valid route
                return None
                
            state_action_alt_repeat.append(next_state)
            cur_state = next_state
        
        # Create and return the new route
        new_route = Route(state_action_alt_repeat, 1, self.pickup_node)
        return new_route

    def _route_to_tuple(self, route):
        """Convert a route to a hashable tuple for checking if already added."""
        return tuple(route.node_in_ordered)

    def add_route(self, route):
        """
        Add a new route to the problem by updating mappings.
        The Xpress model will be rebuilt from scratch using these mappings.
        
        Parameters:
        route (Route): Route to add
        
        Returns:
        int: Index of the new variable
        """
        # Check if this route already exists
        route_tuple = tuple(route.node_in_ordered) if hasattr(route, 'node_in_ordered') else None
        if route_tuple in self.node_sequence_of_routes:
            print(f"Route {route_tuple} already exists in the model")
            # Find the existing route index
            route_idx = self.node_sequence_of_routes.index(route_tuple)
            route_name = ('route', route_idx)
            if route_name in self.route_name_to_index:
                return self.route_name_to_index[route_name]
            else:
                print(f"Warning: Route exists but no variable mapping found")
        
        # Add route to list_of_route and get its index
        route_idx = len(self.list_of_route)
        self.list_of_route.append(route)
        
        # Add route to node sequence if needed
        if route_tuple and route_tuple not in self.node_sequence_of_routes:
            self.node_sequence_of_routes.append(route_tuple)
        
        # Determine variable index - reuse from index_sorted_slot if available
        if hasattr(self, 'index_sorted_slot') and self.index_sorted_slot:
            var_idx = self.index_sorted_slot.pop(0)  # Get the smallest available index
            print(f"Reusing index {var_idx} for new route")
        else:
            # Use next available index
            var_idx = len(self.var_to_obj_coef)
            print(f"Assigned new index {var_idx} for new route")
        
        # Update mappings
        self.var_to_obj_coef[var_idx] = route.cost
        
        # Update constraint coefficient mappings
        exog_vec = route.Exog_vec
        non_zero_indices = np.nonzero(exog_vec)[0]
        
        for row in non_zero_indices:
            coef = float(exog_vec[row])
            self.row_to_col_to_data[row][var_idx] = coef
        
        # Update route mappings
        self.index_to_route_name[var_idx] = ('route', route_idx)
        self.route_name_to_index[('route', route_idx)] = var_idx
        
        print(f"Added route with variable index {var_idx}, route index {route_idx}")
        print(f"Route cost: {route.cost}")
        if hasattr(route, 'just_nodes_ordered'):
            print(f"Route nodes: {route.just_nodes_ordered}")
        
        # Return the index of the new variable
        return var_idx
        
    def remove_doi(self, u, v):
        """
        Remove a DOI (Degree of Interest) omega variable from the mappings.
        The actual Xpress model will be rebuilt separately using rebuild_xpress_model().
        
        Parameters:
        u (int): From node
        v (int): To node
        
        Returns:
        bool: True if successful, False otherwise
        """
        # Check if this omega variable exists
        omega_name = ('omega', u, v)
        if omega_name not in self.omega_name_to_index:
            print(f"Warning: Omega variable for pair ({u},{v}) does not exist")
            return False
            
        # Get the variable index
        var_idx = self.omega_name_to_index[omega_name]
        
        try:
            # Remove from constraint coefficient mappings
            for row in range(len(self.rhs_exog_vec)):
                if row in self.row_to_col_to_data and var_idx in self.row_to_col_to_data[row]:
                    del self.row_to_col_to_data[row][var_idx]
            
            # Remove from objective coefficient mapping
            if var_idx in self.var_to_obj_coef:
                del self.var_to_obj_coef[var_idx]
            
            # Remove from omega mappings
            del self.index_to_omega_name[var_idx]
            del self.omega_name_to_index[omega_name]
            
            # Add index to reuse list, ensuring it exists first
            if not hasattr(self, 'index_sorted_slot'):
                from sortedcontainers import SortedList
                self.index_sorted_slot = SortedList()
            self.index_sorted_slot.add(var_idx)
            
            # If we had the solution values, remove this variable's value
            if hasattr(self, 'var_values') and var_idx in self.var_values:
                del self.var_values[var_idx]
                
            # Decrement omega count
            self.col_of_omega -= 1
            
            # Add this pair to forbidden pairs
            if not hasattr(self, 'forbidden'):
                self.forbidden = set()
            
            self.forbidden.add((u, v))
            
            print(f"Successfully removed omega variable mapping for pair ({u},{v})")
            return True
            
        except Exception as e:
            print(f"Error removing omega variable mapping for pair ({u},{v}): {e}")
            import traceback
            traceback.print_exc()
            return False
    # def remove_route(self, route):
    #     """
    #     Remove a route from the problem.
        
    #     Parameters:
    #     route (Route): The route object to remove
        
    #     Returns:
    #     bool: True if successful, False otherwise
    #     """
        
    #     # Convert route to tuple for lookups
    #     route_tuple = self._route_to_tuple(route)
        
    #     # Check if route is in node_sequence_of_routes
    #     if route_tuple not in self.node_sequence_of_routes:
    #         print(f"Warning: Route {route_tuple} is not found in node_sequence_of_routes")
    #         return False
            
    #     # Find the route index in list_of_route
    #     route_idx = None
    #     for idx, r in enumerate(self.list_of_route):
    #         if self._route_to_tuple(r) == route_tuple:
    #             route_idx = idx
    #             break
                
    #     if route_idx is None:
    #         print(f"Warning: Route not found in list_of_route")
    #         return False
            
    #     # Check if route has a corresponding variable
    #     if route_idx not in self.route_index_to_var_index:
    #         print(f"Warning: Route index {route_idx} has no associated variable")
    #         return False
            
    #     # Get the variable index
    #     var_idx = self.route_index_to_var_index[route_idx]
        
    #     try:
    #         # Get the variable to remove
    #         var_to_remove = self.variables[var_idx]
            
    #         # Remove the variable from the model
    #         self.model.delVar(var_to_remove)
            
    #         # Remove the route from node_sequence_of_routes
    #         self.node_sequence_of_routes.remove(route_tuple)
            
    #         # Remove from tracking dictionaries
    #         if var_idx in self.var_to_obj_coef:
    #             del self.var_to_obj_coef[var_idx]
                
    #         if var_idx in self.var_to_col_coef:
    #             del self.var_to_col_coef[var_idx]
                
    #         # Remove from route mappings
    #         del self.var_index_to_route_index[var_idx]
    #         del self.route_index_to_var_index[route_idx]
            
    #         # Remove from variables dictionary
    #         del self.variables[var_idx]
            
    #         # If we had solution values, remove this variable's value
    #         if hasattr(self, 'var_values') and var_idx in self.var_values:
    #             del self.var_values[var_idx]
                
    #         # Decrement path count
    #         self.col_of_path -= 1
            
    #         # Rebuild objective and constraints after removing the variable
    #         # Note: This depends on the Xpress API capabilities
    #         # If available, update the objective and constraints
            
    #         return True
            
    #     except Exception as e:
    #         print(f"Error removing route {route_tuple}: {e}")
    #         return False
            
    # def add_DOI(self, u, v):
    #     """
    #     Add a DOI (Degree of Interest) variable to the model.
        
    #     Parameters:
    #     u (int): From node
    #     v (int): To node
        
    #     Returns:
    #     int: Index of the new variable, or None if unsuccessful
    #     """
        
    #     # Check if this omega variable already exists
    #     omega_name = ('omega', u, v)
    #     if omega_name in self.omega_name_to_index:
    #         print(f"Warning: Omega variable for pair ({u},{v}) already exists")
    #         return self.omega_name_to_index[omega_name]
            
    #     # Check if the pair (u,v) is forbidden
    #     if (u, v) in self.forbidden:
    #         print(f"Warning: Pair ({u},{v}) is in the forbidden list")
    #         return None
            
    #     # Check if u and v are valid pickup nodes
    #     if u not in self.pickup_node or v not in self.pickup_node:
    #         print(f"Warning: Either {u} or {v} is not a valid pickup node")
    #         return None
            
    #     # Check if v is a neighbor of u
    #     if v not in set(self.neighbors[u]) & set(self.pickup_node):
    #         print(f"Warning: Node {v} is not a neighbor of node {u}")
    #         return None
            
    #     # Create a new variable index
    #     var_idx = max(self.var_to_obj_coef.keys()) + 1 if self.var_to_obj_coef else 0
        
    #     try:
    #         # Calculate the objective coefficient (rho value)
    #         obj_coef = self.rho.get((u, v))
    #         if obj_coef is None:
    #             # If rho[(u,v)] doesn't exist, calculate it
    #             if u != v:
    #                 obj_coef = 2*self.actions[(u,v)][0].cost + 2*self.actions[(u+len(self.pickup_node),v+len(self.pickup_node))][0].cost
    #                 obj_coef = (obj_coef*1.01)+1
    #                 self.rho[(u,v)] = obj_coef
    #             else:
    #                 print(f"Warning: Cannot create omega variable for u=v ({u})")
    #                 return None
                    
    #         # Create the column coefficient vector
    #         col_coef = np.zeros(len(self.pickup_node))
    #         col_coef[u-1] = -1  # Assuming 1-based indexing for pickup nodes
    #         col_coef[v-1] = 1
            
    #         # Create new variable
    #         new_var = xp.var(name=f"omega_{u}_{v}", lb=0)
            
    #         # Add variable to the model
    #         self.model.addVariable(new_var)
            
    #         # Update our tracking dictionaries
    #         self.variables[var_idx] = new_var
    #         self.var_to_obj_coef[var_idx] = obj_coef
    #         self.var_to_col_coef[var_idx] = col_coef
    #         self.omega_name_to_index[omega_name] = var_idx
    #         self.index_to_omega_name[var_idx] = omega_name
            
    #         # Get current objective and update it
    #         try:
    #             current_obj = self.model.getObjective()
    #             new_obj = current_obj + float(obj_coef) * new_var
    #             self.model.setObjective(new_obj)
    #         except Exception as e:
    #             print(f"Warning: Error updating objective when adding DOI: {e}")
    #             # Alternative approach if getObjective is not available
    #             # Rebuild the entire objective from scratch
    #             obj_expr = 0
    #             for v_idx, var in self.variables.items():
    #                 obj_expr += float(self.var_to_obj_coef[v_idx]) * var
    #             self.model.setObjective(obj_expr)
                
    #         # Update all constraints with the new variable
    #         for i, constraint in enumerate(self.constraints):
    #             coef = float(col_coef[i])
    #             if abs(coef) > 1e-10:  # Only add non-zero coefficients
    #                 try:
    #                     # Add term to existing constraint
    #                     constraint_expr = constraint.getExpr()
    #                     new_expr = constraint_expr + coef * new_var
    #                     constraint.setExpr(new_expr)
    #                 except Exception as e:
    #                     print(f"Warning: Error updating constraint {i} when adding DOI: {e}")
    #                     # If direct constraint modification isn't available, would need to
    #                     # rebuild the entire model (not implemented here)
            
    #         # Increment omega count
    #         self.col_of_omega += 1
            
    #         return var_idx
            
    #     except Exception as e:
    #         print(f"Error adding omega variable for pair ({u},{v}): {e}")
    #         return None
        
    

    