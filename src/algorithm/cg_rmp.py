import numpy as np
from collections import defaultdict
import pulp
from src.common.pgm_approach import Route
import itertools
from src.common.state import State
class CG_RMP:
    def __init__(self, list_of_route:list[Route],node_sequence_of_routes, rhs_exog_vec,data, forbidden = [],initial_resource_vector=None):
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
                if u!= v :
                    this_rho[(u,v)] = 2*self.actions[(u,v)][0].cost + 2*self.actions[(u+len(self.pickup_node),v+len(self.pickup_node))][0].cost
                    this_rho[(u,v)]=(this_rho[(u,v)]*1.01)+1
        return this_rho

    def get_forbidden_omega(self):
        forbidden = []
        for var_index, omega_name in self.index_to_omega_name.items():
            if abs(self.var_values[var_index]) > 0.00001:
                u = omega_name[1]
                v = omega_name[2]
                forbidden.append((u,v))
        return forbidden
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
        """Construct the master problem with the generated columns."""
        # Initialize the model
        self.model = pulp.LpProblem("Column_Generation_RMP", pulp.LpMinimize)
        
        # Create decision variables
        for var_idx in self.var_to_obj_coef.keys():
            self.variables[var_idx] = pulp.LpVariable(f"x_{var_idx}", lowBound=0)
        
        #print(f"Created {len(self.variables)} variables")
        
        # Set the objective function - explicitly using Python floats
        obj_expr = 0
        for var_idx, var in self.variables.items():
            obj_expr += float(self.var_to_obj_coef[var_idx]) * var
        
        self.model += obj_expr
        
        # Add constraints - keep the default PuLP naming to ensure compatibility
        constraint_count = 0
        for i in range(len(self.rhs_exog_vec)):
            # Let PuLP handle constraint naming (usually _C1, _C2, etc.)
            self.model += (
                pulp.lpSum([float(self.var_to_col_coef[var_idx][i]) * self.variables[var_idx] 
                           for var_idx in self.variables.keys()]) >= float(self.rhs_exog_vec[i])
            )
            constraint_count += 1
        
        #print(f"Created {constraint_count} constraints")
        #print(f"Constraint names: {list(self.model.constraints.keys())}")
    
    def solve(self):
        """Solve the current master problem."""
        if not self.model:
            raise ValueError("Master problem not initialized.")
        
        # Disable presolve to prevent constraint elimination
        solver = pulp.PULP_CBC_CMD(msg=True, presolve=False)
        
        self.model.solve(solver)
        
        # Return the solution status and objective value
        status = pulp.LpStatus[self.model.status]
        objective_value = float(pulp.value(self.model.objective))
        
        # Get variable values and print them for debugging
        self.var_values = {}
        if self.model.status == pulp.LpStatusOptimal:
            #print("Solution:")
            for var_idx, var in self.variables.items():
                var_value = pulp.value(var)
                self.var_values[var_idx] = var_value
                #print(f"Variable {var_idx}: {var_value}")
        else:
            print(self.model.status)
            input('problem not solve')
            
        # Get dual values
        dual_values = self._get_dual_values()
        
        return {
            'status': status,
            'objective_value': objective_value,
            'variable_values': self.var_values,
            'dual_values': dual_values
        }
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
    
    def solve_ilp(self):

        var_index_to_route_index = defaultdict()
        var_to_obj_coef =defaultdict()
        var_to_col_coef = defaultdict()
        var_index = 0
        for pi in range(len(self.list_of_route)):
            this_path = self.list_of_route[pi]
            var_to_obj_coef[var_index] = this_path.cost
            var_to_col_coef[var_index] = this_path.Exog_vec
            var_index_to_route_index[var_index] = pi
            var_index += 1

        model = pulp.LpProblem("Column_Generation_RMP", pulp.LpMinimize)
        variables = {}
        # Create decision variables
        for var_idx in var_to_obj_coef.keys():
            variables[var_idx] = pulp.LpVariable(f"x_{var_idx}", lowBound=0,cat=pulp.LpInteger)
        
        #print(f"Created {len(self.variables)} variables")
        
        # Set the objective function - explicitly using Python floats
        obj_expr = 0
        for var_idx, var in variables.items():
            obj_expr += float(var_to_obj_coef[var_idx]) * var
        
        model += obj_expr
        
        # Add constraints - keep the default PuLP naming to ensure compatibility
        constraint_count = 0
        for i in range(len(self.rhs_exog_vec)):
            # Let PuLP handle constraint naming (usually _C1, _C2, etc.)
            model += (
                pulp.lpSum([float(var_to_col_coef[var_idx][i]) * variables[var_idx] 
                           for var_idx in variables.keys()]) >= float(self.rhs_exog_vec[i])
            )
            constraint_count += 1
        solver = pulp.PULP_CBC_CMD(msg=True, presolve=True)
       
        model.solve(solver)

        status = pulp.LpStatus[model.status]
        objective_value = pulp.value(model.objective)
        
        # Get variable values and print them for debugging
        var_values = {}
        route_used = []
        if model.status == pulp.LpStatusOptimal:
            #print("Solution:")
            for var_idx, var in variables.items():
                var_value = pulp.value(var)
                var_values[var_idx] = var_value
                if var_value > 0.1:
                    route_used.append(self.list_of_route[var_index_to_route_index[var_idx]])
                #print(f"Variable {var_idx}: {var_value}")
        else:
            print(model.status)
            input('problem not solve')
        
        return {
            'status': status,
            'objective_value': objective_value,
            'variable_values': var_values,
            'used_routes':route_used
        }
    def _get_dual_values(self):
        """Extract dual values from the solved model."""
        if self.model.status != pulp.LpStatusOptimal:
            return None
        
        # Get all constraint names from the model
        constraint_names = list(self.model.constraints.keys())
        #print(f"Available constraint names: {constraint_names}")
        
        dual_values = []
        
        # PuLP uses 1-indexed constraint names like _C1, _C2, etc.
        for i in range(len(self.rhs_exog_vec)):
            # Try to match with PuLP's default naming convention (_C1, _C2, etc.)
            constraint_name = f"_C{i+1}"
            
            if constraint_name in self.model.constraints:
                pi_value = self.model.constraints[constraint_name].pi
                dual_values.append(pi_value)
                #print(f"Found dual value for constraint {i}: {pi_value}")
            else:
                # If not found, try to find by position in the constraints dictionary
                if i < len(constraint_names):
                    alternative_name = constraint_names[i]
                    pi_value = self.model.constraints[alternative_name].pi
                    dual_values.append(pi_value)
                    #print(f"Found dual value using alternative name {alternative_name}: {pi_value}")
                else:
                    dual_values.append(None)
                    #print(f"Warning: Couldn't find dual value for constraint {i}")
        
        return dual_values
        
    def add_column(self, column_coef, obj_coef):

        """Add a new column to the master problem."""
        var_idx = max(self.var_to_obj_coef.keys()) + 1 if self.var_to_obj_coef else 0
        
        # Add column coefficients and objective coefficient
        self.var_to_col_coef[var_idx] = column_coef
        self.var_to_obj_coef[var_idx] = obj_coef
        
        # Create new variable
        self.variables[var_idx] = pulp.LpVariable(f"x_{var_idx}", lowBound=0)
        var = self.variables[var_idx]
        
        # Update the objective function
        self.model += float(obj_coef) * var
        
        # Update the constraints - need to match the naming convention used in the model
        constraint_names = list(self.model.constraints.keys())
        for i in range(len(self.rhs_exog_vec)):
            if i < len(constraint_names):
                constraint_name = constraint_names[i]
                self.model.constraints[constraint_name] += float(column_coef[i]) * var
        
        return var_idx
    

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
        #my_routes = set()  # Keep track of routes we've already added
         
        # Repeat until no more columns to add or max iterations reached
        col_added = 0
        debug_on=False
        cur_obj = np.inf
        #obj_func = self.model.objective
        while num_iter_left > 0:
            obj_func = self.model.objective
            print('check here for loop')
            # Step 3: Solve current RMP
            solution = self.solve()
            
            #input('this lp')

            x_values = solution['variable_values']
            
            # If solution is not optimal, break
            if solution['status'] != 'Optimal':
                break
            this_forbidden_omega = self.get_forbidden_omega()
            if not this_forbidden_omega:
                break
                
            # Step 4: Compute mutation scores for all relevant (l,u,v) triples
            all_mut_scores = []
            my_dual_vals=self._get_dual_values()
            # For each route l with x_l > 0
            for var_idx, x_val in x_values.items():
                if var_idx in self.var_index_to_route_index and x_val > 0.00001:
                    route_idx = self.var_index_to_route_index[var_idx]
                    route = self.list_of_route[route_idx]
                    if len(route.node_in_ordered) <4:
                        continue
                    route_nodes = [node for node in route.node_in_ordered if node in self.pickup_node]
                    
                    # For each u in l
                    for u in route_nodes:
                        # For each v not in l where omega_uv > 0
                        for v in self.pickup_node:
                            if v not in route_nodes:
                                # Check if omega_uv > 0
                                #input('hihi')
                                if ('omega', u, v) in self.omega_name_to_index:
                                    omega_idx = self.omega_name_to_index[('omega', u, v)]
                                    omega_val = x_values.get(omega_idx, 0)
                                    
                                    if omega_val > 0.00001:
                                        # Calculate mutation score
                                        l_hat = self._swap(route, u, v)  # Create new route by swapping u with v
                                        
                                        #print('l_hat')
                                        #print('[u,v]')
                                        ##print([u,v])
                                        #print('route.just_nodes_ordered')
                                        #print(route.just_nodes_ordered)
                                        #print('type(route)')
                                        #print(type(route))
                                        
                                        #print('type(l_hat)')
                                        #print(type(l_hat))
                                        ###input('lookzy')
                                        if l_hat:
                                            this_red_cost=l_hat.get_red_cost(my_dual_vals)
                                            print('this_red_cost')
                                            print(this_red_cost)
                                            #input('this_red_cost')
                                            # Calculate cost difference
                                            cost_l_hat = l_hat.cost
                                            cost_l = route.cost
                                            rho_uv = self.rho[(u, v)]
                                            
                                            mut_score = cost_l_hat - cost_l + rho_uv
                                            
                                            # Add to all mutation scores
                                            #all_mut_scores.append((mut_score, l_hat, route_idx))
                                            all_mut_scores.append(tuple([this_red_cost,l_hat,route_idx]))
            #print('all_mut_scores')
            #print(all_mut_scores)
            #print('above all bb')
            #input('---')

            #for my_tup in self.omega_name_to_index:
            #    my_nm=my_tup[0]
            #    if my_nm=='omega':
            #        u=my_tup[1]#('omega', u, v)
            #        v=my_tup[2]
            #        omega_idx = self.omega_name_to_index[('omega', u, v)]
            #        omega_val = x_values.get(omega_idx, 0)
                    #if omega_val>0.001:
                        #print('uv')
                        #print([u,v])
                        #print('omega_val')
                        #print(omega_val)
            
            # Step 5: Select columns with negative mutation scores
            #cols_to_add = [(score, route, orig_idx) for score, route, orig_idx in all_mut_scores if score <= 0]
            cols_to_add = [(score, route, orig_idx) for score, route, orig_idx in all_mut_scores if score <= -.0001]
            
            # Step 6: Remove routes that are already in my_routes
            for score, route, orig_idx in cols_to_add:
                if route.node_in_ordered in self.node_sequence_of_routes:
                    input('error here for self.node_sequence_of_routes')
            # cols_to_add = [(score, route, orig_idx) for score, route, orig_idx in cols_to_add 
            #             if self._route_to_tuple(route) not in my_routes]
            
            # Step 7: Take subset with smallest scores, up to max_add columns
            cols_to_add.sort(key=lambda x: x[0])  # Sort by mutation score (smallest first)
            cols_to_add = cols_to_add[:max_add]
            
            # If no columns to add, break
            if not cols_to_add:
                break
            #print('cols_to_add')
            #print(cols_to_add)
            #print('input')
            # Step 8: Add selected columns to my_routes and to the problem
            obj_func = self.model.objective
            if debug_on==True:
                this_sol = self.solve()
                this_obj_val=this_sol['objective_value']
                before_obj = this_obj_val
                if this_obj_val<.001:
                    print('error here')
                    input('just before additions lp')

            for _, route, _ in cols_to_add:
                # Add route to my_routes set
                
                
                # Add route to the problem
                var_idx = self._add_route(route)
                print('var_idx')
                print(var_idx)
                print('self.var_to_obj_coef[var_idx]')
                print(self.var_to_obj_coef[var_idx])
                print('route.just_nodes_ordered')
                print(route.just_nodes_ordered)
                col_added +=1
                #input('did ad route')
            # Step 9: Decrement iterations counter
            num_iter_left -= 1
            if debug_on==True:
                this_sol = self.solve()
                print('this_sol')
                print(this_sol)
                this_obj_val=this_sol['objective_value']
                if this_obj_val > before_obj:
                    input('error here: obj not improving with new col')
                var_index_to_value = this_sol['variable_values']
                path_index_used = [self.var_index_to_route_index[idx] for idx in self.var_index_to_route_index.keys() if var_index_to_value[idx]>0.0001]
                path_used = [self.list_of_route[idx] for idx in path_index_used]
                if this_obj_val<.001:
                    print('HOW CAN COST DROP TO ZERO UPON ADDING THESE TERMS')
                print('just after  additions lp')

        # Solve one final time with all the added columns
        final_solution = self.solve()
        
        #input('final lp')
        return final_solution, self.list_of_route, self.node_sequence_of_routes

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
        # Create a valid route
        #try:
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
        print('zip(new_nodes[:-1], new_nodes[1:])')
        print()
        tmp=zip(new_nodes[:-1], new_nodes[1:])
        print('len(new_nodes)')
        print(len(new_nodes))
        
        #input('looking here')
        num_steps=0
        #for (tail, head) in zip(new_nodes[:-1], new_nodes[1:]):
        for i in range(0,len(new_nodes)-1):
            #print('hihihffafa')
            tail=new_nodes[i]
            head=new_nodes[i+1]
            #print('check1')

            this_act = self.actions[(tail, head)][0]
            #print('check2')

            state_action_alt_repeat.append(this_act)
            #print('check3')
            #this_act.pretty_print_action()
            #cur_state.pretty_print_state()
            next_state = this_act.get_head_state_fast_load_ai(cur_state,cur_state.l_id)
            #print('num_steps')
            #print(i)
            if next_state is None:
                # Not a valid route
                #input('route violated')
                return None
            state_action_alt_repeat.append(next_state)
            cur_state = next_state
        
        # Create and return the new route
        new_route = Route(state_action_alt_repeat, 1, self.pickup_node)
        
        
        #input('route created')

        return new_route
        #except:
            # In case of any error, return None
        #    return None

    def _route_to_tuple(self, route):
        """Convert a route to a hashable tuple for checking if already added."""
        return tuple(route.node_in_ordered)

    def _add_route(self, route):
        """
        Add a new route to the problem.
        
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
        new_var = pulp.LpVariable(f"x_{var_idx}", lowBound=0)
        self.variables[var_idx] = new_var
 
        
        # Update the objective function
        self.model.addVariable(new_var)
        self.model.objective += float(route.cost) * new_var
        
        # Update the constraints
        constraint_names = list(self.model.constraints.keys())
        for i in range(len(self.rhs_exog_vec)):
            if i < len(constraint_names):
                constraint_name = constraint_names[i]
                coef = float(route.Exog_vec[i])
                if abs(coef) > 1e-10:  # Only add non-zero coefficients
                    # Get the constraint
                    constraint = self.model.constraints[constraint_name]

                    # Update the constraint by adding the new term directly
                    # This modifies the constraint's expression
                    constraint.addInPlace(coef * new_var)
                    print('check here')
        
        return var_idx