import numpy as np
from typing import List, Dict, Tuple, Any
import networkx as nx
import random

class GWOPricingSolver:
    def __init__(self, graph: nx.DiGraph, max_res: List[float], min_res: List[float], 
                 pop_size: int = 10, max_iter: int = 30):
        self.graph = graph
        self.max_res = np.array(max_res)
        self.min_res = np.array(min_res)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.path = None
        self.total_cost = float('inf')
        self.consumed_resources = None
        
        # Precompute graph data
        self.action_nodes = [n for n in self.graph.nodes() if str(n).startswith('action_')]
        self.normal_nodes = [n for n in self.graph.nodes() if not str(n).startswith('action_')]
        self.edge_weights = {}
        self.edge_resources = {}
        self.node_actions = {}  # Maps nodes to their associated action objects
        
        # Cache graph data
        for u, v, data in self.graph.edges(data=True):
            self.edge_weights[(u, v)] = data['weight']
            self.edge_resources[(u, v)] = data['res_cost']
            if 'action' in data and data['action'] is not None:
                self.node_actions[v] = data['action']
    def check_path_feasibility(self, path: List[str]) -> Tuple[bool, List[str]]:
        """
        Check path feasibility including elementarity constraint
        """
        if not path or len(path) < 2:
            return False, []
            
        resources = np.zeros(len(self.max_res))
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
                if any(new_resources > self.max_res) or any(new_resources < self.min_res):
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
    
    def generate_valid_path(self) -> List[str]:
        """Generate elementary path with feasibility checking"""
        max_attempts = 10
        best_path = None
        best_cost = float('inf')
        
        for _ in range(max_attempts):
            path = ['Source']
            current = 'Source'
            visited = {'Source'}  # Track visited nodes for elementarity
            resources = np.zeros(len(self.max_res))
            
            while current != 'Sink':
                neighbors = []
                for neighbor in self.graph.neighbors(current):
                    edge = (current, neighbor)
                    
                    # Skip if this is a normal node we've already visited
                    if not str(neighbor).startswith('action_') and neighbor != 'Sink' and neighbor in visited:
                        continue
                        
                    # Check if adding this edge would maintain feasibility
                    new_resources = resources + self.edge_resources[edge]
                    if all(new_resources <= self.max_res) and all(new_resources >= self.min_res):
                        neighbors.append((neighbor, self.edge_weights[edge]))
                        
                if not neighbors:
                    break
                    
                # Prefer neighbors with lower reduced cost
                neighbors.sort(key=lambda x: x[1])
                
                # Weighted random selection favoring better costs
                weights = np.exp([-n[1] for n in neighbors])  # Convert costs to weights
                weights = weights / np.sum(weights)  # Normalize
                next_node = np.random.choice([n[0] for n in neighbors], p=weights)
                
                path.append(next_node)
                current = next_node
                
                # Only add normal nodes to visited set (not action nodes)
                if not str(next_node).startswith('action_'):
                    visited.add(next_node)
                    
                resources += self.edge_resources[(path[-2], path[-1])]
                
                if len(path) > 2 * len(self.graph.nodes):
                    break
                    
            if current == 'Sink':
                cost, _ = self.calculate_fitness(path)
                if cost < best_cost:
                    best_path = path
                    best_cost = cost
                    
        return best_path if best_path else ['Source', 'Sink']
    
    def calculate_fitness(self, path: List[str]) -> Tuple[float, np.ndarray]:
        """Calculate fitness with feasibility check"""
        is_feasible, action_nodes = self.check_path_feasibility(path)
        if not is_feasible:
            return float('inf'), np.zeros(len(self.max_res))
            
        reduced_cost = 0
        resources = np.zeros(len(self.max_res))
        
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            reduced_cost += self.edge_weights[edge]
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
                print(f"At iteration {iteration} found new best reduced cost: {best_reduced_cost}")
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
        """Generate path with improved feasibility checking"""
        max_attempts = 10
        best_path = None
        best_cost = float('inf')
        
        for _ in range(max_attempts):
            path = ['Source']
            current = 'Source'
            visited = {'Source'}
            resources = np.zeros(len(self.max_res))
            
            while current != 'Sink':
                neighbors = []
                for neighbor in self.graph.neighbors(current):
                    edge = (current, neighbor)
                    # Check if adding this edge would maintain feasibility
                    new_resources = resources + self.edge_resources[edge]
                    if all(new_resources <= self.max_res) and all(new_resources >= self.min_res):
                        neighbors.append((neighbor, self.edge_weights[edge]))
                        
                if not neighbors:
                    break
                    
                # Prefer neighbors with lower reduced cost
                neighbors.sort(key=lambda x: x[1])
                next_node = neighbors[0][0] if random.random() < 0.7 else random.choice(neighbors)[0]
                
                path.append(next_node)
                current = next_node
                visited.add(next_node)
                resources += self.edge_resources[(path[-2], path[-1])]
                
                if len(path) > 2 * len(self.graph.nodes):
                    break
                    
            if current == 'Sink':
                cost, _ = self.calculate_fitness(path)
                if cost < best_cost:
                    best_path = path
                    best_cost = cost
                    
        return best_path if best_path else ['Source', 'Sink']
    


def modify_generalized_absolute_pricing(pricer, dual_vector: np.ndarray):
    """Modified version of generalized_absolute_pricing using GWO instead of BiDirectional"""
    # Original graph construction code remains the same until solver creation
    index_to_resource = list(pricer.initial_resource_state.keys())
    graph = nx.DiGraph(directed=True, n_res=len(index_to_resource) + 1, elementary=False)
    
    # Build graph (same as original)
    for (origin_node, destination_node), action_list in pricer.actions.items():
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
    max_res = [max_arbitrary_monotone_resource] + [float(pricer.initial_resource_state[res])
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

    list_of_customer = pricer.get_customer_from_path(path)
    #print(list_of_customer)
    beta_term = pricer.generate_beta_term(list_of_customer[1:-1])

    state_set = pricer.generate_state_based_on_beta(beta_term)
    

    return state_set, actions_in_path, total_cost
