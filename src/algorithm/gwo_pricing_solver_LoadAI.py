import numpy as np
from typing import List, Dict, Tuple, Any
import networkx as nx
from networkx import DiGraph
import random
from scipy.sparse import csr_matrix
from src.algorithm.update_states.LoadAI_state_generation_input import LoadAI_state_input
class GWOPricingSolverLoadAI:
    def __init__(self, actions,initial_resource_state,nodes, resource_name_to_index,initial_resource_vector,jy_option,loadAI_input:LoadAI_state_input):
        self.actions = actions
        self.initial_resource_state = initial_resource_state
        self.nodes = nodes
        self.resource_name_to_index = resource_name_to_index
        self.initial_resource_vector = initial_resource_vector
        self.path = None
        self.total_cost = float('inf')
        self.consumed_resources = None
        self.number_of_resources = len(initial_resource_state)
        self.min_resource_state = np.zeros(self.number_of_resources)
        self.initial_resource_vector = initial_resource_vector
        # Precompute graph data
        self.edge_weights = {}
        self.edge_resources = {}
        self.node_actions = {}  # Maps nodes to their associated action objects
        self.pop_size: int = 10 
        self.max_iter: int = 30
        self.jy_option = jy_option
        self.loadAI_input = loadAI_input
        self.time_window_start = loadAI_input.time_window_start
        self.time_window_end = loadAI_input.time_window_end
        self.pickup_node = loadAI_input.pickup_node
        self.dropoff_node = loadAI_input.dropoff_node
        self.problem_info = loadAI_input.problem_info
        self.pickup_to_dropoff = loadAI_input.pickup_to_dropoff
        self.dropoff_to_pickup = loadAI_input.dropoff_to_pickup
        random.seed(1000)
    def initial_resource(self):
        resources = np.array([1])
        resources = np.concatenate([
            resources,
            [self.problem_info['weight_capacity']],
            [self.problem_info['volume_capacity']],
            [np.inf],
            [self.problem_info['max_combined_loads']],
            np.ones(len(self.pickup_node)),
            np.ones(len(self.dropoff_node))
        ])
        return resources
    def check_path_feasibility(self, path: List[str]) -> Tuple[bool, List[str]]:
        """
        Check path feasibility including elementarity constraint
        """
        if not path or len(path) < 2:
            return False, []
            
        resources = self.initial_resource()
        action_nodes = []
        visited = set()  # Track visited normal nodes
        
        for i in range(len(path)):
            current = path[i]
            
            # Check elementarity for normal nodes
            if not str(current).startswith('action_') and current not in ['Source', 'Sink']:
                if current in visited:
                    return False, []  # Non-elementary path
                visited.add(current)
            
            # Check resource constraints
            if i < len(path) - 1:
                edge = (path[i], path[i + 1])
                if edge not in self.edge_resources:
                    return False, []
                    
                new_resources = resources + self.edge_resources[edge]
                if current == 'Source':
                    current = -1
                if current == 'Sink':
                    current = -2 
                if any(new_resources > self.max_res[current]) or any(new_resources < self.min_res[current]):
                    return False, []
                    
                resources = new_resources
                
                if str(path[i + 1]).startswith('action_'):
                    action_nodes.append(path[i + 1])
                    
        return True, action_nodes              
        
    def initialize_population(self):
        """Initialize wolf population with valid paths"""
        population = []
        for _ in range(self.pop_size):
            path = self.generate_valid_path()
            fitness = self.calculate_fitness(path)
            population.append((path, fitness))
        return population
    
    
    def calculate_fitness(self, path: List[str]) -> Tuple[float, np.ndarray]:
        """Calculate fitness with feasibility check"""
        # is_feasible, action_nodes = self.check_path_feasibility(path)
        # if not is_feasible:
        #     return float('inf'), np.zeros(len(self.max_res))
            
        reduced_cost = 0
        resources = self.initial_resource()
        
        if path ==['Source','Sink']:
            reduced_cost = np.inf
            resources = np.inf
        else:
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                try:
                    reduced_cost += self.edge_weights[edge]
                except:
                    print('check here')
                resources += self.edge_resources[edge]
            
        return reduced_cost, resources
    
    def update_position(self, current_pos: List[str], alpha_pos: List[str], 
                    beta_pos: List[str], delta_pos: List[str], a: float) -> List[str]:
        """Update position while maintaining elementarity"""
        if random.random() < a:  # Exploration phase
            return self.generate_valid_path()
            
        # Exploitation phase - learn from leaders while keeping elementarity
        leaders = [alpha_pos, beta_pos, delta_pos]
        new_path = ['Source']
        current = 'Source'
        visited = {'Source'}  # Track visited nodes
        
        while current != 'Sink':
            # Get next nodes from leaders' paths
            next_nodes = []

            for leader in leaders:
                try:
                    idx = leader.index(current)
                    if idx < len(leader) - 1:
                        next_node = leader[idx + 1]
                        # Only add if it maintains elementarity
                        if str(next_node).startswith('action_') or next_node == 'Sink' or next_node not in visited:
                            next_nodes.append(next_node)
                except ValueError:
                    continue
            
            # If no guidance from leaders, use available neighbors
            if not next_nodes:
                neighbors = []
                for neighbor in self.graph.neighbors(current):
                    if str(neighbor).startswith('action_') or neighbor == 'Sink' or neighbor not in visited:
                        neighbors.append(neighbor)
                        
                if not neighbors:
                    return self.generate_valid_path()
                next_node = random.choice(neighbors)
            else:
                next_node = random.choice(next_nodes)
                
            new_path.append(next_node)
            current = next_node
            
            # Update visited set for elementarity
            if not str(next_node).startswith('action_') and next_node != 'Sink':
                visited.add(next_node)
                
            if len(new_path) > len(self.graph.nodes):
                return self.generate_valid_path()
                
        return new_path
        
    def run(self):
        """Execute GWO with improved feasibility handling"""
        population = []
        for _ in range(self.pop_size):
            path = self.generate_valid_path()
            fitness = self.calculate_fitness(path)
            if fitness[0] != float('inf'):  # Only add feasible solutions
                population.append((path, fitness))
                
        if not population:
            self.path = None
            return
            
        best_reduced_cost = float('inf')
        stagnation_counter = 0
        
        for iteration in range(self.max_iter):
            # Sort by reduced cost
            population.sort(key=lambda x: x[1][0])
            current_best = population[0][1][0]
            
            # Update best solution
            if current_best < best_reduced_cost:
                best_reduced_cost = current_best
                self.path = population[0][0]
                self.total_cost = current_best
                self.consumed_resources = population[0][1][1]
                stagnation_counter = 0
                #print(f"At iteration {iteration} found new best reduced cost: {best_reduced_cost}")
            else:
                stagnation_counter += 1
                
            # Early stopping
            if best_reduced_cost < -0.1 and stagnation_counter > 10:
                break
                
            # GWO updates
            alpha = population[0][0]
            beta = population[1][0] if len(population) > 1 else alpha
            delta = population[2][0] if len(population) > 2 else beta
            
            a = 2 * (1 - iteration / self.max_iter)
            
            # Generate new positions and maintain feasibility
            new_population = []
            for _ in range(self.pop_size):
                new_pos = self.update_position(population[0][0], alpha, beta, delta, a)
                fitness = self.calculate_fitness(new_pos)
                if fitness[0] != float('inf'):  # Only add feasible solutions
                    new_population.append((new_pos, fitness))
                    
            # Keep best solutions from both populations
            population = sorted(population + new_population,
                             key=lambda x: x[1][0])[:self.pop_size]


    def generate_valid_path(self) -> List[str]:
        """Generate path with improved feasibility checking using beam search width=2"""
        max_attempts = 10
        best_path = None
        best_cost = float('inf')
        
        for attempt in range(max_attempts):
            # Initialize with 2 identical starting states
            states = [
                {
                    'path': ['Source'],
                    'current': 'Source',
                    'visited': {'Source'},
                    'avoid_cycle': [],
                    'can_drop_off': [],
                    'resources': self.initial_resource(),
                    'cost': 0
                }
            ]
            
            iteration = 0
            max_iterations = 4 * len(self.graph.nodes)  # Safety limit
            
            # Keep expanding states until we reach the sink or run out of states
            while states and iteration < max_iterations:
                iteration += 1
                next_states = []
                
                # For each current state, find all possible next nodes
                for state in states:
                    current = state['current']
                    
                    # Already reached sink - add to next_states and continue
                    if current == 'Sink':
                        next_states.append(state)
                        continue
                    
                    path = state['path']
                    visited = state['visited']
                    avoid_cycle = state['avoid_cycle']
                    can_drop_off = state['can_drop_off']
                    resources = state['resources']
                    
                    neighbors = []
                    
                    # Find all possible neighbors, exactly as in the original code
                    for neighbor in self.graph.neighbors(current):
                        edge = (current, neighbor)
                        
                        # First check - compute temporary resources
                        temp_resources = resources.copy() + self.edge_resources[edge]
                        
                        if isinstance(neighbor, str) and neighbor != 'Sink':
                            parts = neighbor.split('_')
                            des_node = int(parts[2])
                            
                            if des_node not in avoid_cycle:
                                if des_node in self.pickup_node:
                                    # Keep original time constraint handling
                                    temp_resources[self.resource_name_to_index['time']+1] = min(
                                        temp_resources[self.resource_name_to_index['time']+1],
                                        self.time_window_start[des_node]
                                    )
                                    if all(temp_resources <= self.max_res[des_node]) and all(temp_resources >= self.min_res[des_node]):
                                        neighbors.append((neighbor, self.edge_weights[edge]))
                                        
                                if des_node in self.dropoff_node and des_node in can_drop_off:
                                    # Keep original time constraint handling
                                    temp_resources[self.resource_name_to_index['time']+1] = min(
                                        temp_resources[self.resource_name_to_index['time']+1],
                                        self.time_window_start[des_node]
                                    )
                                    if all(temp_resources <= self.max_res[des_node]) and all(temp_resources >= self.min_res[des_node]):
                                        neighbors.append((neighbor, self.edge_weights[edge]))
                                        
                                if des_node == -2:
                                    neighbors.append((neighbor, self.edge_weights[edge]))
                                    
                        elif isinstance(neighbor, int):
                            neighbors.append((neighbor, self.edge_weights[edge]))
                            
                        elif neighbor == 'Sink':
                            des_node = -2
                            temp_resources[self.resource_name_to_index['time']+1] = min(
                                temp_resources[self.resource_name_to_index['time']+1],
                                self.time_window_start[des_node]
                            )
                            if all(temp_resources <= self.max_res[des_node]) and all(temp_resources >= self.min_res[des_node]):
                                neighbors.append((neighbor, self.edge_weights[edge]))
                    
                    if not neighbors:
                        continue  # No valid neighbors from this state
                    
                    # Sort neighbors by edge weight (like original)
                    neighbors.sort(key=lambda x: x[1])
                    
                    # Try at most 2 neighbors from each state
                    selected_neighbors = []
                    
                    # Always consider the best neighbor
                    selected_neighbors.append(neighbors[0])
                    
                    # Add a random neighbor with probability 0.7 (like original)
                    if len(neighbors) > 1 and random.random() < 0.7:
                        # Pick a random neighbor that's not already selected
                        candidates = [n for n in neighbors[1:]]
                        if candidates:
                            selected_neighbors.append(random.choice(candidates))
                    
                    # Create a new state for each selected neighbor
                    for next_node, edge_weight in selected_neighbors:
                        new_path = path + [next_node]
                        new_visited = visited.copy()
                        new_visited.add(next_node)
                        
                        # Update resources EXACTLY as in original code
                        new_resources = resources.copy() + self.edge_resources[(path[-1], next_node)]
                        new_avoid_cycle = avoid_cycle.copy()
                        new_can_drop_off = can_drop_off.copy()
                        
                        # Apply resource constraints based on node type
                        if isinstance(next_node, int):
                            new_resources[self.resource_name_to_index['time']+1] = min(
                                new_resources[self.resource_name_to_index['time']+1],
                                self.time_window_start[next_node]
                            )
                            new_avoid_cycle.append(next_node)
                            if next_node in self.pickup_node:
                                new_can_drop_off.append(self.pickup_to_dropoff[next_node])
                        
                        # Calculate cost
                        new_cost = state['cost'] + edge_weight
                        
                        new_state = {
                            'path': new_path,
                            'current': next_node,
                            'visited': new_visited,
                            'avoid_cycle': new_avoid_cycle,
                            'can_drop_off': new_can_drop_off,
                            'resources': new_resources,
                            'cost': new_cost
                        }
                        
                        next_states.append(new_state)
                
                # If no more valid next states, end this attempt
                if not next_states:
                    break
                    
                # Sort by cost and keep at most 2 best states
                next_states.sort(key=lambda s: s['cost'])
                states = next_states[:min(10, len(next_states))]
                
                # If both states reach the sink, we can stop
                if all(state['current'] == 'Sink' for state in states):
                    break
                    
            # Check if we found any paths to the sink
            completed_states = [s for s in states if s['current'] == 'Sink']
            
            for state in completed_states:
                cost, _ = self.calculate_fitness(state['path'])
                if cost < best_cost:
                    best_path = state['path']
                    best_cost = cost
        
        # Return the best path found or default path
        if best_path:
            return best_path
        else:
            # Fallback to the original implementation for one final attempt
            original_path = self._original_generate_valid_path()
            if original_path and original_path != ['Source', 'Sink']:
                return original_path
            return ['Source', 'Sink']
    


    def construct_graph(self,dual_vector: np.ndarray):
        """Modified version of generalized_absolute_pricing using GWO instead of BiDirectional"""
        # Original graph construction code remains the same until solver creation
        index_to_resource = list(self.initial_resource_state.keys())
        graph = nx.DiGraph(directed=True, n_res=len(index_to_resource) + 1, elementary=False)
        ARBITRARY_MONOTONE_RESOURCE_CONSUMPTION = float(1)
        ARBITRARY_RESOURCE_MAX = float(10000)
        # Build graph (same as original)
        node_need = self.pickup_node | self.dropoff_node| {-1,-2}
        for (origin_node, destination_node), action_list in self.actions.items():
            if origin_node in node_need and destination_node in node_need:
                if origin_node ==4 and destination_node==6:
                    print('check here')
                for action in action_list:
                    if origin_node == -1:
                        origin_node = "Source"
                    if destination_node == -2:
                        destination_node = "Sink"
                        
                    action_node = f"action_{action.node_tail}_{action.node_head}_{action.action_id}"

                    exog_duals = dual_vector[:len(action.Exog_vec)]
                    dual_contribution = np.dot(action.Exog_vec, exog_duals)
                    edge_weight = action.cost - dual_contribution
                    
                    arbitrary_monotone_resource_consumption = float(1)
                    first_array = np.array(ARBITRARY_MONOTONE_RESOURCE_CONSUMPTION).reshape(-1)
                    second_array = action.resource_consumption_vec.toarray()[0] 
                    this_res_cost = np.concatenate([first_array,second_array])
                    third_array = np.array(ARBITRARY_MONOTONE_RESOURCE_CONSUMPTION).reshape(-1)
                    this_res_cost_2 = np.concatenate([third_array, np.zeros(self.number_of_resources)])
                    #print(origin_node,destination_node,this_res_cost)
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
        self.max_res = {}
        
        self.min_res={}
        for node in self.nodes:
            if node in (self.pickup_node | self.dropoff_node | {-1,-2}):
                this_max_res = [ARBITRARY_RESOURCE_MAX] + list(self.initial_resource_vector.toarray()[0])
                this_max_res[self.resource_name_to_index['time']+1] = self.time_window_start[node]
                self.max_res[node] = this_max_res
                this_min_res = [0]+ list(self.min_resource_state)
                this_min_res[self.resource_name_to_index['time']+1] = self.time_window_end[node]
                self.min_res[node] = this_min_res
        return graph
    def call_gwo_pricing(self,dual):
        self.graph:DiGraph = self.construct_graph(dual)
        self.action_nodes = [n for n in self.graph.nodes() if str(n).startswith('action_')]
        self.normal_nodes = [n for n in self.graph.nodes() if not str(n).startswith('action_')]
        # Cache graph data
        for u, v, data in self.graph.edges(data=True):
            self.edge_weights[(u, v)] = data['weight']
            self.edge_resources[(u, v)] = data['res_cost']
            if 'action' in data and data['action'] is not None:
                self.node_actions[v] = data['action']
    
        # Use GWO instead of BiDirectional
        self.run()
        
        path = self.path
        total_cost = self.total_cost
        resources_used = self.consumed_resources
        actions_in_path = []
        states = []
        
        # Process results (same as original)
        #print(path)
        if path and total_cost < -1e-6:
            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i + 1]]
                action = edge_data.get("action")
                if action is not None:
                    actions_in_path.append(action)
            
            #path[0] = -1
            #path[-1] = -2
            #print(f"this is the weird path {path}")
        # states = pricer.get_states_from_action_list_new(actions_in_path)
        # print(states)
        #print(path)
        list_of_nodes, list_of_actions = self._get_nodes_and_actions_from_path(path, self.graph)
        

        return list_of_nodes, list_of_actions, total_cost
    def _get_nodes_and_actions_from_path(self,path: list, graph: nx.DiGraph):
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