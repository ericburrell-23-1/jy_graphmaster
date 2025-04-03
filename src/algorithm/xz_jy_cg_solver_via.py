import numpy as np
from collections import defaultdict
from src.common.pgm_approach import Route
import itertools
from src.common.state import State
import xpress as xp

class xy_jy_cg_solver:
    def __init__(self, list_of_route:list[Route], node_sequence_of_routes, rhs_exog_vec, data, forbidden=[], initial_resource_vector=None):
        self.list_of_route = list_of_route
        self.list_of_action_list = []
        self.node_sequence_of_routes = node_sequence_of_routes
        self.rhs_exog_vec = rhs_exog_vec
        self.var_to_obj_coef = defaultdict()
        self.var_to_col_coef = defaultdict()
        self.variables = {}

        self.actions = data.actions

        self.pickup_node = data.pickup_node
        self.dropoff_node = data.dropoff_node
        self.neighbors = data.neighbors
        self.forbidden = forbidden
        self.initial_resource_vector = initial_resource_vector
        self.rho = self._generate_rho()
        self._generate_col_coeff()
        
        self._build_master_problem()
        self.time_profile = defaultdict(float)

    def _generate_rho(self):
        this_rho = defaultdict()
        for u in self.pickup_node:
            for v in self.pickup_node:
                if u!= v:
                    this_rho[(u,v)] = 2*self.actions[(u,v)][0].cost + 2*self.actions[(u+len(self.pickup_node),v+len(self.pickup_node))][0].cost
                    this_rho[(u,v)]=(this_rho[(u,v)]*1.01)+1
        return this_rho
        
    def _generate_col_coeff(self):
        var_index = 0
        self.col_of_path = 0
        self.col_of_omega = 0
        self.omega_name_to_index = defaultdict()
        self.index_to_omega_name = defaultdict()
        self.var_index_to_route_index = defaultdict()
        self.route_index_to_var_index = defaultdict()
        for pi in range(len(self.list_of_route)):
            route = self.list_of_route[pi]
            self.var_to_obj_coef[var_index] = route.cost
            self.var_to_col_coef[var_index] = route.Exog_vec
            self.var_index_to_route_index[var_index] = pi
            self.route_index_to_var_index[pi] = var_index
            var_index += 1
            self.col_of_path += 1
        for u in self.pickup_node:
            for v in set(self.neighbors[u]) & set(self.pickup_node):
                if (u,v) not in self.forbidden:
                    self.var_to_obj_coef[var_index] = self.rho[(u,v)]
                    vec = np.zeros(len(self.pickup_node))
                    vec[u-1] = -1
                    vec[v-1] = 1
                    self.var_to_col_coef[var_index] = vec
                    self.omega_name_to_index[('omega',u,v)] = var_index
                    self.index_to_omega_name[var_index] = ('omega',u,v)
                    var_index += 1
                    self.col_of_omega +=1
            
    def _build_master_problem(self):
        """Construct the master problem with the generated columns using Xpress."""
        
        # Initialize the model
        self.model = xp.problem()
        
        # Create decision variables
        self.variables = {}
        for var_idx in self.var_to_obj_coef.keys():
            self.variables[var_idx] = xp.var(name=f"x_{var_idx}", lb=0)
        
        # Set the objective function
        obj_expr = xp.Sum(float(self.var_to_obj_coef[var_idx]) * self.variables[var_idx] 
                          for var_idx in self.variables.keys())
        self.model.setObjective(obj_expr, sense=xp.minimize)
        
        # Add constraints
        self.constraints = []
        for i in range(len(self.rhs_exog_vec)):
            constraint_expr = xp.Sum(float(self.var_to_col_coef[var_idx][i]) * self.variables[var_idx] 
                                     for var_idx in self.variables.keys())
            constraint = constraint_expr >= float(self.rhs_exog_vec[i])
            self.model.addConstraint(constraint, name=f"c_{i}")
            self.constraints.append(constraint)

    def generate_subset_of_routes(self):
        sub_set_routes = []
        for route in self.list_of_route:
            node_in_ordered = route.node_in_ordered
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
    
    def solve(self):
        """Solve the current master problem using Xpress."""
        
        if not hasattr(self, 'model'):
            raise ValueError("Master problem not initialized.")
        
        # Set solver parameters
        self.model.setControl({
            'presolve': 0,  # Disable presolve
            'outputlog': 1,  # Enable output logging
            'lpiterlimit': 10000,  # Set a reasonable iteration limit
            'lprefactfrequency': 100  # Re-factorize basis after this many iterations
        })
        
        # Solve the problem
        self.model.reset()
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
        cur_obj = np.inf
        
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
            my_dual_vals = self.grab_dual_sol()
            
            # For each route l with x_l > 0
            for var_idx, x_val in x_values.items():
                if var_idx in self.var_index_to_route_index and x_val > 0.00001:
                    route_idx = self.var_index_to_route_index[var_idx]
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
                                if ('omega', u, v) in self.omega_name_to_index:
                                    omega_idx = self.omega_name_to_index[('omega', u, v)]
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
            for score, route, orig_idx in cols_to_add:
                if route.node_in_ordered in self.node_sequence_of_routes:
                    input('error here for self.node_sequence_of_routes')
            
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
                    input('just before additions lp')

            for _, route, _ in cols_to_add:
                # Add route to the problem
                var_idx = self.add_route(route)
                print('var_idx')
                print(var_idx)
                print('self.var_to_obj_coef[var_idx]')
                print(self.var_to_obj_coef[var_idx])
                print('route.just_nodes_ordered')
                print(route.just_nodes_ordered)
                col_added += 1
                
            # Step 9: Decrement iterations counter
            num_iter_left -= 1
            
            if debug_on == True:
                this_sol = self.solve()
                print('this_sol')
                print(this_sol)
                this_obj_val = this_sol['objective_value']
                if this_obj_val > before_obj:
                    input('error here: obj not improving with new col')
                var_index_to_value = this_sol['variable_values']
                path_index_used = [self.var_index_to_route_index[idx] for idx in self.var_index_to_route_index.keys() if var_index_to_value[idx] > 0.0001]
                path_used = [self.list_of_route[idx] for idx in path_index_used]
                if this_obj_val < .001:
                    print('HOW CAN COST DROP TO ZERO UPON ADDING THESE TERMS')
                print('just after additions lp')

        # Solve one final time with all the added columns
        final_solution = self.solve()
        
        return final_solution, self.list_of_route, self.node_sequence_of_routes
    def solve_ilp(self):
        """Solve the integer version of the problem using Xpress."""
        
        var_index_to_route_index = defaultdict()
        var_to_obj_coef = defaultdict()
        var_to_col_coef = defaultdict()
        var_index = 0
        for pi in range(len(self.list_of_route)):
            this_path = self.list_of_route[pi]
            var_to_obj_coef[var_index] = this_path.cost
            var_to_col_coef[var_index] = this_path.Exog_vec
            var_index_to_route_index[var_index] = pi
            var_index += 1

        # Initialize the model
        model = xp.problem()
        variables = {}
        
        # Create decision variables - integer
        for var_idx in var_to_obj_coef.keys():
            variables[var_idx] = xp.var(name=f"x_{var_idx}", lb=0, vartype=xp.integer)
        
        # Set the objective function
        obj_expr = xp.Sum(float(var_to_obj_coef[var_idx]) * variables[var_idx] 
                         for var_idx in variables.keys())
        model.setObjective(obj_expr, sense=xp.minimize)
        
        # Add constraints
        for i in range(len(self.rhs_exog_vec)):
            constraint_expr = xp.Sum(float(var_to_col_coef[var_idx][i]) * variables[var_idx] 
                                     for var_idx in variables.keys())
            model.addConstraint(constraint_expr >= float(self.rhs_exog_vec[i]), name=f"c_{i}")
        
        # Set solver parameters
        model.setControl({
            'presolve': 1,  # Enable presolve for ILP
            'outputlog': 1  # Enable output logging
        })
        
        # Solve the problem
        model.solve()
        
        # Get solution status
        status_code = model.getProbStatus()
        if status_code == xp.mip_optimal:
            status = "Optimal"
        else:
            status = f"Other ({status_code})"
        
        # Get objective value
        objective_value = model.getObjVal() if status == "Optimal" else float('inf')
        
        # Get variable values and used routes
        var_values = {}
        route_used = []
        if status == "Optimal":
            for var_idx, var in variables.items():
                var_value = model.getSolution(var)
                var_values[var_idx] = var_value
                if var_value > 0.1:
                    route_used.append(self.list_of_route[var_index_to_route_index[var_idx]])
        else:
            print(model.getProbStatus())
            input('problem not solve')
        
        return {
            'status': status,
            'objective_value': objective_value,
            'variable_values': var_values,
            'used_routes': route_used
        }
    def grab_primal_sol(self):
        return self.objective_value, self.var_values
    def grab_dual_sol(self):
        """Extract dual values from the solved model."""
        
        status_code = self.model.getProbStatus()
        if status_code != xp.lp_optimal:
            return None
        
        dual_values = []
        
        # Get dual values for each constraint
        for i in range(len(self.rhs_exog_vec)):
            constraint_name = f"c_{i}"
            try:
                # Get dual value (shadow price) for this constraint
                dual_value = self.model.getDual(constraint_name)
                dual_values.append(dual_value)
            except:
                dual_values.append(None)
                print(f"Warning: Couldn't find dual value for constraint {i}")
        
        return dual_values
    
    

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
        
        print('u')
        print(u)
        print('v')
        print(v)
        print('nodes')
        print(nodes)
        print('new_nodes')
        print(new_nodes)
        print('len(new_nodes)')
        print(len(new_nodes))
        
        for i in range(0, len(new_nodes)-1):
            tail = new_nodes[i]
            head = new_nodes[i+1]

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
        Add a new route to the problem using Xpress.
        
        Parameters:
        route (Route): Route to add
        
        Returns:
        int: Index of the new variable
        """
        
        # Add route to list_of_route
        route_idx = len(self.list_of_route)
        self.node_sequence_of_routes.append(self._route_to_tuple(route))
        self.list_of_route.append(route)
        
        # Create new variable
        var_idx = max(self.var_to_obj_coef.keys()) + 1 if self.var_to_obj_coef else 0
        
        # Add column coefficients and objective coefficient
        self.var_to_obj_coef[var_idx] = route.cost
        self.var_to_col_coef[var_idx] = route.Exog_vec
        self.var_index_to_route_index[var_idx] = route_idx
        self.route_index_to_var_index[route_idx] = var_idx
        
        # Create new variable
        new_var = xp.var(name=f"x_{var_idx}", lb=0)
        self.variables[var_idx] = new_var
        
        # Prepare column data: pairs of (constraint_index, coefficient)
        column_data = []
        for i in range(len(self.rhs_exog_vec)):
            coef = float(route.Exog_vec[i])
            if abs(coef) > 1e-10:  # Only include non-zero coefficients
                column_data.append((f"c_{i}", coef))
        
        # Add the variable with its coefficients to the model
        self.model.addVariable(new_var)
        
        # Add objective coefficient
        self.model.setObjectiveCoeff(new_var, float(route.cost))
        
        # Add coefficients to constraints
        for constraint_name, coef in column_data:
            constraint = self.model.getConstraint(constraint_name)
            self.model.addTerm(constraint, new_var, coef)
        
        return var_idx
    def remove_route(self, route):
        """
        Remove a route from the problem.
        
        Parameters:
        route (Route): The route object to remove
        
        Returns:
        bool: True if successful, False otherwise
        """
        
        # Convert route to tuple for lookups
        route_tuple = self._route_to_tuple(route)
        
        # Check if route is in node_sequence_of_routes
        if route_tuple not in self.node_sequence_of_routes:
            print(f"Warning: Route {route_tuple} is not found in node_sequence_of_routes")
            return False
            
        # Find the route index in list_of_route
        route_idx = None
        for idx, r in enumerate(self.list_of_route):
            if self._route_to_tuple(r) == route_tuple:
                route_idx = idx
                break
                
        if route_idx is None:
            print(f"Warning: Route not found in list_of_route")
            return False
            
        # Check if route has a corresponding variable
        if route_idx not in self.route_index_to_var_index:
            print(f"Warning: Route index {route_idx} has no associated variable")
            return False
            
        # Get the variable index
        var_idx = self.route_index_to_var_index[route_idx]
        
        try:
            # Get the variable to remove
            var_to_remove = self.variables[var_idx]
            
            # Remove the variable from the model
            self.model.delVar(var_to_remove)
            
            # Remove the route from node_sequence_of_routes
            self.node_sequence_of_routes.remove(route_tuple)
            
            # Remove from tracking dictionaries
            if var_idx in self.var_to_obj_coef:
                del self.var_to_obj_coef[var_idx]
                
            if var_idx in self.var_to_col_coef:
                del self.var_to_col_coef[var_idx]
                
            # Remove from route mappings
            del self.var_index_to_route_index[var_idx]
            del self.route_index_to_var_index[route_idx]
            
            # Remove from variables dictionary
            del self.variables[var_idx]
            
            # If we had solution values, remove this variable's value
            if hasattr(self, 'var_values') and var_idx in self.var_values:
                del self.var_values[var_idx]
                
            # Decrement path count
            self.col_of_path -= 1
            
            return True
            
        except Exception as e:
            print(f"Error removing route {route_tuple}: {e}")
            return False
    def add_DOI(self,u,v):
        
        # Check if this omega variable already exists
        omega_name = ('omega', u, v)
        if omega_name in self.omega_name_to_index:
            print(f"Warning: Omega variable for pair ({u},{v}) already exists")
            return self.omega_name_to_index[omega_name]
            
        # Check if the pair (u,v) is forbidden
        if (u, v) in self.forbidden:
            print(f"Warning: Pair ({u},{v}) is in the forbidden list")
            return None
            
        # Check if u and v are valid pickup nodes
        if u not in self.pickup_node or v not in self.pickup_node:
            print(f"Warning: Either {u} or {v} is not a valid pickup node")
            return None
            
        # Check if v is a neighbor of u
        if v not in set(self.neighbors[u]) & set(self.pickup_node):
            print(f"Warning: Node {v} is not a neighbor of node {u}")
            return None
            
        # Create a new variable index
        var_idx = max(self.var_to_obj_coef.keys()) + 1 if self.var_to_obj_coef else 0
        
        try:
            # Calculate the objective coefficient (rho value)
            obj_coef = self.rho.get((u, v))
            if obj_coef is None:
                # If rho[(u,v)] doesn't exist, calculate it
                if u != v:
                    obj_coef = 2*self.actions[(u,v)][0].cost + 2*self.actions[(u+len(self.pickup_node),v+len(self.pickup_node))][0].cost
                    obj_coef = (obj_coef*1.01)+1
                    self.rho[(u,v)] = obj_coef
                else:
                    print(f"Warning: Cannot create omega variable for u=v ({u})")
                    return None
                    
            # Create the column coefficient vector
            col_coef = np.zeros(len(self.pickup_node))
            col_coef[u-1] = -1  # Assuming 1-based indexing for pickup nodes
            col_coef[v-1] = 1
            
            # Create new variable
            new_var = xp.var(name=f"omega_{u}_{v}", lb=0)
            
            # Add variable to the model
            self.model.addVariable(new_var)
            
            # Update the objective function
            self.model.addToObj(float(obj_coef) * new_var)
            
            # Update the constraints
            for i in range(len(self.rhs_exog_vec)):
                constraint_name = f"c_{i}"
                coef = float(col_coef[i])
                if abs(coef) > 1e-10:  # Only add non-zero coefficients
                    self.model.addCoefficient(constraint_name, new_var, coef)
            
            # Update our tracking dictionaries
            self.variables[var_idx] = new_var
            self.var_to_obj_coef[var_idx] = obj_coef
            self.var_to_col_coef[var_idx] = col_coef
            self.omega_name_to_index[omega_name] = var_idx
            self.index_to_omega_name[var_idx] = omega_name
            
            # Increment omega count
            self.col_of_omega += 1
            
            return var_idx
            
        except Exception as e:
            print(f"Error adding omega variable for pair ({u},{v}): {e}")
            return None
        
    def remove_doi(self, u, v):
        
        # Check if this omega variable exists
        omega_name = ('omega', u, v)
        if omega_name not in self.omega_name_to_index:
            print(f"Warning: Omega variable for pair ({u},{v}) does not exist")
            return False
            
        # Get the variable index
        var_idx = self.omega_name_to_index[omega_name]
        
        try:
            # Get the variable to remove
            var_to_remove = self.variables[var_idx]
            
            # Remove the variable from the model
            self.model.delVar(var_to_remove)
            
            # Remove from our tracking dictionaries
            if var_idx in self.var_to_obj_coef:
                del self.var_to_obj_coef[var_idx]
                
            if var_idx in self.var_to_col_coef:
                del self.var_to_col_coef[var_idx]
                
            # Remove from omega mappings
            del self.index_to_omega_name[var_idx]
            del self.omega_name_to_index[omega_name]
            
            # Remove from variables dictionary
            del self.variables[var_idx]
            
            # If we had the solution values, remove this variable's value
            if hasattr(self, 'var_values') and var_idx in self.var_values:
                del self.var_values[var_idx]
                
            # Decrement omega count
            self.col_of_omega -= 1
            
            return True
            
        except Exception as e:
            print(f"Error removing omega variable for pair ({u},{v}): {e}")
            return False

    def get_active_DOI(self):
        forbidden = []
        for var_index, omega_name in self.index_to_omega_name.items():
            if abs(self.var_values[var_index]) > 0.00001:
                u = omega_name[1]
                v = omega_name[2]
                forbidden.append((u,v))
        return forbidden