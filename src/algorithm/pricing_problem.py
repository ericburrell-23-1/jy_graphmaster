import os
import random
import networkx as nx
import numpy as np
from cspy import BiDirectional, REFCallback
from typing import List, Dict, Tuple
from src.common.action import Action
from src.common.state import State
from src.common.helper import Helper
from gwo_pricing_solver import GWOPricingSolver
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

    def __init__(self, actions: Dict[Tuple[int, int], List[Action]], initial_resource_state: Dict[str, int], nodes: List[int], resource_name_to_index: Dict[str, int],initial_resource_vector:np.ndarray):
        self.actions = actions
        self.initial_resource_state = initial_resource_state
        self.nodes = nodes
        self.number_of_resources = len(initial_resource_state)
        self.min_resource_state = np.zeros(self.number_of_resources)
        self.resource_name_to_index = resource_name_to_index
        self.initial_resource_vector = initial_resource_vector
        #self._compute_resource_limits()
        
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

    def gwo_pricing(self,dual_vector:np.np.ndarray):
        """Modified version of generalized_absolute_pricing using GWO instead of BiDirectional"""
        # Original graph construction code remains the same until solver creation
        index_to_resource = list(self.initial_resource_state.keys())
        graph = nx.DiGraph(directed=True, n_res=len(index_to_resource) + 1, elementary=False)
        
        # Build graph (same as original)
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
                
                arbitrary_monotone_resource_consumption = float(1)
                
                graph.add_edge(
                    origin_node,
                    action_node,
                    res_cost=np.array([arbitrary_monotone_resource_consumption] + 
                                    [-float(action.trans_term_vec[resource])
                                    for resource in index_to_resource]),
                    weight=edge_weight,
                    action=action
                )
                
                graph.add_edge(
                    action_node,
                    destination_node,
                    res_cost=np.concatenate([[arbitrary_monotone_resource_consumption], 
                                        np.zeros(len(index_to_resource))]),
                    weight=0,
                    action=None
                )
        
        max_arbitrary_monotone_resource = float(1000000)
        min_arbitrary_monotone_resource = float(0)
        max_res = [max_arbitrary_monotone_resource] + [float(self.initial_resource_state[res])
                                                    for res in index_to_resource]
        min_res = [min_arbitrary_monotone_resource] + [0.0] * (len(index_to_resource))
        
        # Use GWO instead of BiDirectional
        solver = GWOPricingSolver(graph, max_res=max_res, min_res=min_res)
        solver.run()
        
        path = solver.path
        total_cost = solver.total_cost
        resources_used = solver.consumed_resources
        actions_in_path = []
        states = []
        
        # Process results (same as original)
        #print(path)
        if path and total_cost < -1e-6:
            for i in range(len(path) - 1):
                edge_data = graph[path[i]][path[i + 1]]
                action = edge_data.get("action")
                if action is not None:
                    actions_in_path.append(action)
            
            #path[0] = -1
            #path[-1] = -2
            #print(f"this is the weird path {path}")
        # states = pricer.get_states_from_action_list_new(actions_in_path)
        # print(states)
        #print(path)

        list_of_customer = self._get_nodes_and_actions_from_path(graph,path)


        return list_of_customer, actions_in_path, total_cost

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
        ARBITRARY_RESOURCE_MAX = float(10000)
        graph = nx.DiGraph(directed=True, n_res=self.number_of_resources + 1, elementary=False)

        for (origin_node, destination_node), action_list in self.actions.items():
            for action in action_list:
                if origin_node == -1:
                    origin_node = "Source"

                if destination_node == -2:
                    destination_node = "Sink"

                action_node = f"action_{action.node_tail}_{action.node_head}_{action.action_id}"

                exog_duals = dual_vector[:len(action.Exog_vec)]
                dual_contribution = np.dot(action.Exog_vec, exog_duals)
                edge_weight = action.cost - dual_contribution

                # Create edge from `origin_node` to `action_node`
                first_array = np.array(ARBITRARY_MONOTONE_RESOURCE_CONSUMPTION).reshape(-1)
                second_array = action.resource_consumption_vec.toarray()[0] * -1
                this_res_cost = np.concatenate([first_array,second_array])
                third_array = np.array(ARBITRARY_MONOTONE_RESOURCE_CONSUMPTION).reshape(-1)
                this_res_cost_2 = np.concatenate([third_array, np.zeros(self.number_of_resources)])
                print(origin_node,destination_node,this_res_cost)
                graph.add_edge(
                    origin_node,
                    action_node,
                    res_cost = this_res_cost,
                    # res_cost=np.array([ARBITRARY_MONOTONE_RESOURCE_CONSUMPTION] + [-float(action.trans_term_add[resource])
                    #                   for resource in self.initial_resource_state]),
                    weight=edge_weight,

                    action=action  # Store the action object for traceability
                )

                # Create edge from `action_node` to `destination_node`
                graph.add_edge(
                    action_node,
                    destination_node,
                    res_cost=this_res_cost_2,
                    weight=0,
                    action=None
                )
        max_res = [ARBITRARY_RESOURCE_MAX] + list(self.initial_resource_vector.toarray()[0])
        print('max_res',max_res)
        #max_res = self.max_resource_state
        min_res = [0]+ list(self.min_resource_state)
        print('min_res',min_res)
        #min_res = self.min_resource_state

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
