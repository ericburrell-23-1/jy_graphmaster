from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Set
from numpy import ndarray, array
from src.common.action import Action
from src.common.state import State
from src.algorithm.solver import GraphMaster
from src.algorithm.update_states.state_update_function import StateUpdateFunction


class OptimizationProblem(ABC):
    def __init__(self, problem_instance_file_name: str, file_type: str):
        self.problem_instance_file_name: str = problem_instance_file_name
        self.file_type: str = file_type
        self.nodes: List[int] = []
        self.rhs_vector: ndarray = array([])
        self.initial_resource_state: Dict[str, int] = {}
        self.actions: Dict[Tuple[int, int], Action] = {}
        self.initial_res_states: Set[State] = set()
        self.initial_res_actions: Set[Action] = set()
        self.state_update_module: StateUpdateFunction = None
                
        self._load_data_from_file()   
        self._build_problem_model()
        self._create_initial_res_states()
        self._create_initial_res_actions()
        self._define_state_update_module()

    @abstractmethod
    def solve(self):
        """Creates a GraphMasterSolver instance from problem data and calls its solve() method"""

        self.solver = GraphMaster(
            self.actions,
            self.rhs_vector,
            self.nodes,
            self.initial_resource_state,
            self.initial_res_states,
            self.initial_res_actions,
            self.state_update_module
        )
        self.solver.solve()

    @abstractmethod
    def _load_data_from_file(self):
        """Load problem-specific data from the file named in `problem_instance_file_name`."""
        pass


    @abstractmethod
    def _build_problem_model(self):
        """Creates Nodes, Actions, RHS, and R_Init for each problem type."""
        pass


    @abstractmethod
    def _create_initial_res_states(self):
        """Build initial `res_states`, which forms the initial feasible solution for the solver."""
        pass


    @abstractmethod
    def _create_initial_res_actions(self):
        """Build initial `res_actions`, which forms the initial feasible solution for the solver when using PGM."""
        pass

    @abstractmethod
    def _define_state_update_module(self):
        """Defines the `state_update_module`, which is used to update `res_states` after pricing is finished."""