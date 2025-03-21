import numpy as np
from collections import defaultdict
import pulp

class CG_RMP:
    def __init__(self, list_of_action_list, rhs_exog_vec):
        self.list_of_action_list = list_of_action_list
        self.rhs_exog_vec = rhs_exog_vec
        self.var_to_obj_coef = defaultdict()
        self.var_to_col_coef = defaultdict()
        self.variables = {}
        self.model = None
        self._geenrate_col_coeff()
        self._build_master_problem()

    def _geenrate_col_coeff(self):
        var_index = 0
        for pi in range(len(self.list_of_action_list)):
            this_cover = np.zeros(len(self.rhs_exog_vec))
            this_obj_coef = 0
            for a in self.list_of_action_list[pi]:
                this_cover += a.Exog_vec
                this_obj_coef += a.cost
            self.var_to_obj_coef[var_index] = this_obj_coef
            self.var_to_col_coef[var_index] = this_cover
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
        var_values = {}
        if self.model.status == pulp.LpStatusOptimal:
            #print("Solution:")
            for var_idx, var in self.variables.items():
                var_value = pulp.value(var)
                var_values[var_idx] = var_value
                #print(f"Variable {var_idx}: {var_value}")
        else:
            print(self.model.status)
            input('problem not solve')
            
        # Get dual values
        dual_values = self._get_dual_values()
        
        return {
            'status': status,
            'objective_value': objective_value,
            'variable_values': var_values,
            'dual_values': dual_values
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