from typing import Dict, Set, List, Tuple, Any
import xpress as xp
import numpy as np
from src.common.state import State
from src.common.action import Action
from src.common.multi_state_graph import MultiStateGraph
from src.common.variable import Variable
from src.common.constraint import Constraint
from scipy.sparse import coo_matrix


class GraphMasterRestrictedMasterProblem:
    """
    IBuilds the GraphMaster RMP. Also provides a `solve` method to call the LP solver.
    """

    def __init__(self, multi_graph: MultiStateGraph, rhs_exog_vec: Dict[int, float]):
        """
        Initialize with multigraph and RHS values.

        Args:
            multi_graph: MultiStateGraph containing states, actions, and classes
            rhs_exog_vec: Right-hand side values for constraints
        """
        self.variable_name_to_variable : Dict[Any,Variable] = {}
        self.constraint_name_to_constraint:Dict[Any,Constraint] = {}

        self.multi_graph = multi_graph
        self.rhs_exog_vec = rhs_exog_vec
        self.problem = xp.problem()

        self._form_restricted_master_problem()

    def add_variable(self, name, lower_bound, upper_bound, obj_coef):
        """Adds a variable to variable dictionary"""
        self.variable_name_to_variable[name] = Variable(
            name, lower_bound, upper_bound, obj_coef, False)

    def add_constraint(self, name, lower_bound, upper_bound, var_val_dict):
        """Adds a constraint to constraints dictionary"""
        self.constraint_name_to_constraint[name] = Constraint(
            name, lower_bound, upper_bound, var_val_dict)

    def init_variables(self):
        """Initiate variable dictionary"""
        pass

    def init_constraints(self):
        """Initiate constarints dictionary"""
        pass


    def _form_restricted_master_problem(self):
        """Build the Restricted Master Problem from scratch for the first instance."""
        pass

    def solve_restricted_master_problem(self):
        """Solve the problem and return duals"""
        pass

    def print_solution(self):
        """Function used to easily display the solution to the problem. Intended for debugging and verifying results."""
        pass