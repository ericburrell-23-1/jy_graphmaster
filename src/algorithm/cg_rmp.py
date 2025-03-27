import numpy as np
from collections import defaultdict
import pulp
from src.common.pgm_approach import Route
import itertools
from src.common.state import State

class CG_RMP:
    def __init__(self, list_of_route:list[Route], rhs_exog_vec,data, forbidden = [],initial_resource_vector=None):
        self.list_of_route = list_of_route
        self.list_of_action_list = []

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
        self._geenrate_col_coeff()
        
        self._build_master_problem()


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
    def _geenrate_col_coeff(self):
        var_index = 0
        self.omega_name_to_index = defaultdict()
        self.index_to_omega_name = defaultdict()
        self.var_index_to_route_index = defaultdict()
        for pi in range(len(self.list_of_route)):
            route = self.list_of_route[pi]
            self.var_to_obj_coef[var_index] = route.cost
            self.var_to_col_coef[var_index] = route.Exog_vec
            self.var_index_to_route_index[var_index] = pi
            var_index += 1
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
        objective_value = pulp.value(self.model.objective)
        
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
            this_route = Route(state_action_alt_repeat,1)
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