import os
import random
import networkx as nx
import numpy as np
from cspy import BiDirectional, REFCallback
from typing import List, Dict, Tuple
from src.common.action import Action
from src.common.state import State
EPSILON = 0.000001

class PricingProblem:
    """
    Represents the pricing subproblem.
    Arguments:
        actions: A dictionary where keys are tuples (origin_node, destination_node) and values are lists of actions.
        initial_resource_state: A dictionary of resource limits for the problem.
    Methods:
        find_lowest_reduced_cost_path: Solves the pricing subproblem.
    """

    def __init__(self, actions: Dict[Tuple[int, int], List[Action]], initial_resource_state: Dict[str, int], nodes: List[int]):
        self.actions = actions
        self.initial_resource_state = initial_resource_state
        self.nodes = nodes

        self._compute_resource_limits()
        
        # self.resource_extension_function = ResourceExtensionCallback(
        #     self.max_resource_state, self.min_resource_state)

    def _compute_resource_limits(self):
        """Computes `min_resource_state` and `max_resource_state` as `np.array` objects, to be used when solving pricing. 
        Note that the first resource is an arbitrary monotone resource, as required by `cspy.BiDirectional`."""
        self.number_of_resources = len(self.initial_resource_state) + 1
        self.min_resource_state = np.zeros(self.number_of_resources)

        # `resource_name_list` helps us preserve the ordering of all resources in `np.array` objects
        self.resource_name_list = [resource_name for resource_name in self.initial_resource_state]

        ARBITRARY_RESOURCE_MAX = float(10000)
        max_resource_state_list = [ARBITRARY_RESOURCE_MAX]
        for resource_name in self.resource_name_list:
            max_resource_state_list.append(self.initial_resource_state[resource_name])

        self.max_resource_state = np.array(max_resource_state_list)


    def generalized_absolute_pricing(self, dual_vector: np.ndarray):
        """
        This is the simplest RCSPP solver that does not make use of Resource Extension Function or DSSR.
        Supports multiple edges between nodes (one for each action).
        Computes the shortest path in a resource-constrained DiGraph and returns the associated actions.

        Arguments:
            dual_vector: A numpy array representing the dual values of all constraints.

        Returns:
            A tuple containing:
            - A list of nodes visited in the shortest path
            - A list of actions corresponding to the edges in the shortest path.
            - The total cost of the path

        Note: this function sucks. It's here so that we just have SOME way to solve pricing, especially for small problems.
        """
        ARBITRARY_MONOTONE_RESOURCE_CONSUMPTION = float(1)
        graph = nx.DiGraph(directed=True, n_res=self.number_of_resources, elementary=False)

        for (origin_node, destination_node), action_list in self.actions.items():
            for action in action_list:
                if origin_node == -1:
                    origin_node = "Source"

                if destination_node == -2:
                    destination_node = "Sink"

                action_node = f"action_{action.origin_node}_{action.destination_node}_{action.action_id}"

                exog_duals = dual_vector[:len(action.contribution_vector)]
                dual_contribution = np.dot(action.contribution_vector, exog_duals)
                edge_weight = action.cost - dual_contribution

                # Create edge from `origin_node` to `action_node`
                graph.add_edge(
                    origin_node,
                    action_node,

                    res_cost=np.array([ARBITRARY_MONOTONE_RESOURCE_CONSUMPTION] + [-float(action.trans_term_vec[resource])
                                      for resource in self.resource_name_list]),
                    weight=edge_weight,

                    action=action  # Store the action object for traceability
                )

                # Create edge from `action_node` to `destination_node`
                graph.add_edge(
                    action_node,
                    destination_node,
                    res_cost=np.concatenate(
                        [[ARBITRARY_MONOTONE_RESOURCE_CONSUMPTION], np.zeros(self.number_of_resources - 1)]),
                    weight=0,
                    action=None
                )

        max_res = self.max_resource_state
        min_res = self.min_resource_state

        # Solve the RCSPP
        problem = BiDirectional(
            graph, max_res=max_res, min_res=min_res, direction="both", elementary=False)
        problem.run()

        path = problem.path
        list_of_nodes, list_of_actions = self._get_nodes_and_actions_from_path(path, graph)
        
        total_cost = problem.total_cost

        return list_of_nodes, list_of_actions, total_cost

    @staticmethod
    def _get_nodes_and_actions_from_path(path: list, graph: nx.DiGraph):
        """Returns `list_of_nodes` and `list_of_actions` found in `path`."""
        list_of_nodes = []
        list_of_actions = []
        if path:
            for i in range(len(path) - 1):
                origin_node = path[i]
                destination_node = path[i + 1]
                edge_data = graph[origin_node][destination_node]
                action = edge_data.get("action")
                if action is not None:
                    list_of_actions.append(action)
                    list_of_nodes.append(origin_node)
        
        list_of_nodes[0] = -1
        list_of_nodes.append(-2)

        return list_of_nodes, list_of_actions













class ResourceExtensionCallback(REFCallback):
    """This is what is used to define REF. I had a lot of trouble getting this to work as intended."""
    def __init__(self, max_resource_state, min_resource_state):
        super().__init__()
        self.graph: nx.DiGraph = None
        self._max_res = max_resource_state
        self._min_res = min_resource_state

    def REF_fwd(self, cumulative_resource, tail, head, edge_resource_consumption, partial_path, accummulated_cost):
        pass

    def REF_bwd(self, cumulative_resource, tail, head, edge_resource_consumption, partial_path, accummulated_cost):
        pass

    def REF_join(self, fwd_resource, bwd_resource, tail, head, edge_resource_consumption):
        pass
