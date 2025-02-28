from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Set
from numpy import ndarray, array
from src.common.action import Action
from src.common.state import State
from src.algorithm.solver import GraphMaster
from src.algorithm.update_states.state_update_function import StateUpdateFunction
from itertools import permutations
from collections import defaultdict

class OptimizationProblem(ABC):
    def __init__(self, problem_instance_file_name: str, file_type: str):
        self.problem_instance_file_name: str = problem_instance_file_name
        self.file_type: str = file_type
        self.nodes: List[int] = []
        self.rhs_vector: ndarray = array([])
        self.rhs_dict: Dict[str, float] = {}
        self.rhs_constraint_name_to_index: Dict[str, int] = {}
        self.rhs_index_to_constraint_name: Dict[int, str] = {}
        self.initial_resource_vector: ndarray = array([])
        self.initial_resource_dict: Dict[str, int] = {}
        self.resource_name_to_index: Dict[str, int] = {}
        self.resource_index_to_name: Dict[int, str] = {}
        self.number_of_resources: int = None
        self.actions: Dict[Tuple[int, int], List[Action]] = {}
        self.initial_res_states: Set[State] = set()
        self.initial_res_actions: Set[Action] = set()
        self.the_single_null_action: Action = None
        self.state_update_module: StateUpdateFunction = None
                
        self._load_data_from_file()   
        self._build_problem_model()
        self._create_dom_action_object()
        self._create_initial_res_states()
        self._create_initial_res_actions()
        self._generate_neighbors()
        self._define_state_update_module()

    @abstractmethod
    def solve(self):
        """Creates a GraphMasterSolver instance from problem data and calls its solve() method"""
        #node_to_list = self._group_states_by_node_l(self.initial_res_states)
        self.solver = GraphMaster(
            self.nodes,
            self.actions,
            self.rhs_vector,
            self.initial_resource_dict,
            self.initial_resource_vector,
            self.initial_res_states,
            self.initial_res_actions,
            self.state_update_module,
            self.dominated_action_pairs,
            self.resource_name_to_index,
            self.number_of_resources,
            self.the_single_null_action,
            
            #node_to_list
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
    def _generate_neighbors(self):
        """generate neighbors"""
        pass
    
    @abstractmethod
    def _create_initial_res_actions(self):
        """Build initial `res_actions`, which forms the initial feasible solution for the solver when using PGM."""
        pass

    @abstractmethod
    def _define_state_update_module(self):
        """Defines the `state_update_module`, which is used to update `res_states` after pricing is finished."""
        pass

    def _create_dom_action_object(self):
        
        dom_actions_pairs=dict()
        for _, action_list in self.actions.items():
            for action1, action2 in permutations(action_list, 2):
                if action1.get_is_dominated(action2):
                    if action1 in dom_actions_pairs:
                        dom_actions_pairs[action1].append(action2)
                    else:
                        dom_actions_pairs[action1] = [action2]

        self.dominated_action_pairs = dom_actions_pairs

    def _group_states_by_node_l(self,resStates):
        """Groups states by (l_id, node) into a dictionary of lists with structure {l_id: {node: [states]}}."""
        dict_l_node_2_list = defaultdict(lambda: defaultdict(list))  # Nested defaultdict for automatic list initialization
    
        # Group states by l_id and node
        for state in resStates:
            dict_l_node_2_list[state.l_id][state.node].append(state)

        # Check that each l_id has exactly one source and one sink
        for l_id in dict_l_node_2_list:
            source_count = len(dict_l_node_2_list[l_id].get(-1, []))
            sink_count = len(dict_l_node_2_list[l_id].get(-2, []))

            if source_count != 1 or sink_count != 1:
                raise ValueError(f"Graph {l_id} must have exactly one source and one sink, but found {source_count} source(s) and {sink_count} sink(s).")
    
        return dict_l_node_2_list
    
    

