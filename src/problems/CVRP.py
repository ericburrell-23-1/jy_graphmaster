
from src.problems.optimization_problem import OptimizationProblem
from src.common.state import State
from src.common.action import Action
from src.algorithm.update_states.standard_CVRP import CVRP_state_update_function
from numpy import zeros, ones
from math import hypot

class CVRP(OptimizationProblem):

    def __init__(self, problem_instance_file_name: str, file_type: str = "Standard_VRP"):
        """Defines all aspects of a CVRP problem needed before calling the `solve` method."""
        super().__init__(problem_instance_file_name, file_type)

    def solve(self):
        super().solve()
        pass
        

    def _load_data_from_file(self):
        """Parses data from file. Handles different types of files depending on `file_type` property."""
        match self.file_type:
            case "Standard_VRP":
                self._load_standard_vrp_file()
            # Call file parsing functions for other file types here
            case _:
                raise Exception(f"No file parsing logic available for file type {self.file_type}")
        

    def _load_standard_vrp_file(self):
        """Loads raw data for `Standard_VRP` file type."""
        file_path = self.problem_instance_file_name
        with open(file_path, "r") as file:
            lines = file.readlines()

        coord_section = False
        demand_section = False
        coordinates = {}
        demands = {}
        capacity = None

        for line in lines:
            if line.startswith("CAPACITY"):
                capacity = int(line.split(":")[1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                coord_section = True
                demand_section = False
                continue
            elif line.startswith("DEMAND_SECTION"):
                demand_section = True
                coord_section = False
                continue
            elif line.startswith("DEPOT_SECTION"):
                break

            if coord_section:
                columns = line.split()
                if len(columns) == 3:
                    customer_id = int(columns[0])
                    x = float(columns[1])
                    y = float(columns[2])
                    if customer_id == 1:
                        coordinates[-1] = coordinates[-2] = (x, y)
                    else:
                        coordinates[customer_id - 1] = (x, y)
                else:
                    print("Error parsing coordinates")

            elif demand_section:
                columns = line.split()
                if len(columns) == 2:
                    customer_id = int(columns[0])
                    demand = int(columns[1])
                    if customer_id == 1:
                        demands[-1] = demands[-2] = demand
                    else:
                        demands[customer_id - 1] = demand
                else:
                    print("Error parsing demand")

        self.capacity = capacity
        self.demands = demands
        self.coordinates = coordinates

    def _build_problem_model(self):
        num_customers = len(self.demands) - 2

        self.nodes = []
        self.initial_resource_state = {"cap_remain": self.capacity}
        self.rhs_vector = ones(num_customers)
        self.actions = {}
        
        idx = 0
        self.constraint_name_to_index = {}
        for node in self.demands.keys():
            self.nodes.append(node)

        for origin_node in self.nodes:
            if origin_node > 0:
                # DEFINE COVERAGE CONSTRAINT RHS
                self.constraint_name_to_index[("Cover", origin_node)] = idx
                idx += 1

            for destination_node in self.nodes:
                if origin_node == destination_node or origin_node==-2 or destination_node == -1:
                    continue
                if origin_node == -1 and destination_node == -2:
                    continue
                cost = self._distance(origin_node, destination_node)
                contribution_vector = zeros(num_customers)
                if origin_node > 0:
                    contribution_vector[self.constraint_name_to_index[("Cover", origin_node)]] = 1
                min_resource_vector = {"cap_remain": self.demands[origin_node] + self.demands[destination_node]}
                resource_consumption_vector = {"cap_remain": -self.demands[origin_node]}
                max_resource_vector = {"cap_remain": float('inf')}
                if origin_node == -1:
                    for u in self.nodes:
                        if u not in (-1,-2):
                            min_resource_vector[f'can_visit: {u}'] = 0
                            resource_consumption_vector[f'can_visit: {u}']=0
                            max_resource_vector[f'can_visit: {u}']=1
                if destination_node == -2:
                    for u in self.nodes:
                        if u not in (-1,-2):
                            if u == origin_node:
                                resource_consumption_vector[f'can_visit: {u}']=-1
                            else:
                                resource_consumption_vector[f'can_visit: {u}']=0
                            min_resource_vector[f'can_visit: {u}'] = 0
                            max_resource_vector[f'can_visit: {u}']=1
                else:
                    for u in self.nodes:
                        if u not in (-1,-2):
                            if u == destination_node:
                                min_resource_vector[f'can_visit: {u}'] = 1
                            else:
                                min_resource_vector[f'can_visit: {u}'] = 0
                            if u == origin_node:
                                resource_consumption_vector[f'can_visit: {u}'] = -1
                            else:
                                resource_consumption_vector[f'can_visit: {u}'] = 0
                            max_resource_vector[f'can_visit: {u}'] = 1
                action = Action(origin_node, destination_node, cost, contribution_vector, min_resource_vector, resource_consumption_vector, max_resource_vector)

                self.actions[origin_node, destination_node] = [action]


    def _create_initial_res_actions(self):
        """Note to Julian: No code exists for this yet."""
        return super()._create_initial_res_actions()
    
    def _create_initial_res_states(self):
        """Note to Julian: No code exists for this yet."""
        return super()._create_initial_res_states()
    
    def _define_state_update_module(self):
        """This is where we define how res_states is updated after pricing. We are using the `standard_CVRP` module for this definition."""
        self.state_update_module = CVRP_state_update_function(self.coordinates, self.demands)


    def _distance(self, origin, destination):
        """Helper function to get costs of actions easily."""
        x1, y1 = self.coordinates[origin]
        x2, y2 = self.coordinates[destination]

        return hypot(x2 - x1, y2 - y1)

