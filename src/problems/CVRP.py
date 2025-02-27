from collections import ChainMap
from src.problems.optimization_problem import OptimizationProblem
from src.common.state import State
from src.common.action import Action
from src.algorithm.update_states.standard_CVRP import CVRP_state_update_function
import numpy as np
from numpy import zeros, ones
from math import hypot
from scipy.sparse import csr_matrix
from src.common.helper import Helper
 
class CVRP(OptimizationProblem):
 
    def __init__(self, problem_instance_file_name: str, file_type: str = "Standard_VRP"):
        """Defines all aspects of a CVRP problem needed before calling the `solve` method."""
        self.neighbors_by_distance = {}
        super().__init__(problem_instance_file_name, file_type)
        self._generate_neighbors()
 
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

        self.nodes = [-1]
        self.initial_resource_dict = {"cap_remain": self.capacity}
        self.rhs_vector = ones(num_customers)
        self.actions = {}
        
        idx = 0
        self.constraint_name_to_index = {}
        for node in self.demands.keys():
            if node not in {-1,-2}:
                self.nodes.append(node)
                self.initial_resource_dict[f'can_visit: {node}'] = 1
        
        self.nodes.append(-2)
 
 
        #Make default dictionary for actions
        self.number_of_resources = num_customers + 1 # number of customer + source + sink + capremain
        self.default_min_resource_vector=np.array([])
        self.default_max_resource_vector=np.array([])
        self.default_resource_consumption_vector=np.array([])
        self.default_min_resource_dict = {}
        self.default_max_resource_dict = {}
        self.default_resource_consumption_dict = {}
        self.default_min_resource_dict["cap_remain"]=0
        self.default_max_resource_dict["cap_remain"]=self.capacity
        self.default_resource_consumption_dict["cap_remain"]=0
        self.resource_name_to_index["cap_remain"] = 0
        self.default_min_resource_vector=np.zeros(self.number_of_resources)
        self.default_max_resource_vector=np.ones(self.number_of_resources)
        self.default_max_resource_vector[self.resource_name_to_index["cap_remain"]]=self.capacity
        #put in
        self.default_resource_consumption_vec=np.zeros(self.number_of_resources)
        
        idx = 1
        for u in self.nodes:
            if u in (-1,-2):
                continue
            self.default_min_resource_dict[f'can_visit: {u}'] = 0
            self.default_max_resource_dict[f'can_visit: {u}'] = 1
            self.default_resource_consumption_dict[f'can_visit: {u}'] = 0
            self.resource_name_to_index[f'can_visit: {u}'] = idx
            idx +=1
        
        self.default_exog_name_to_coeff_dict = {}
        for node in self.nodes:
             self.default_exog_name_to_coeff_dict[("Cover", node)] = 0
 
        idx = 0
        for origin_node in self.nodes:
            if origin_node > 0:
                # DEFINE COVERAGE CONSTRAINT RHS
                self.constraint_name_to_index[str(("Cover", origin_node))] = idx
                self.rhs_constraint_name_to_index[str(("Cover", origin_node))] = idx
                self.rhs_index_to_constraint_name[idx] = str(("Cover", origin_node))
                idx += 1
 
            for destination_node in self.nodes:
                if origin_node == destination_node or origin_node==-2 or destination_node == -1:
                    continue
                if origin_node == -1 and destination_node == -2:
                    continue
                cost = self._distance(origin_node, destination_node)
                contribution_vector = zeros(num_customers)
                if origin_node > 0:
                    contribution_vector[self.constraint_name_to_index[str(("Cover", origin_node))]] = 1
                partial_min_resource_dict = {"cap_remain": self.demands[origin_node] + self.demands[destination_node]}
                partial_max_resource_dict = {}
                partial_resource_consumption_dict = {"cap_remain": -self.demands[origin_node]}
                
                if origin_node != -1:
                    partial_resource_consumption_dict[f'can_visit: {origin_node}'] =-1
                if destination_node!=-2:
                    partial_min_resource_dict[f'can_visit: {destination_node}']= 1
                #str("Cover", origin_node)
 
                #print('origin_node')
                #print(origin_node)
                #print('destination_node')
                #print(destination_node)
                #print('partial_resource_consumption_dict')
                #print(partial_resource_consumption_dict)
                #print('partial_max_resource_dict')
                #print(partial_max_resource_dict)
                #print('partial_min_resource_dict')
                #print(partial_min_resource_dict)
                #input('----')
                trans_min_input = ChainMap({**partial_min_resource_dict, **self.default_min_resource_dict})
                trans_term_min = ChainMap({**partial_max_resource_dict, **self.default_max_resource_dict})
                trans_term_add = ChainMap({**partial_resource_consumption_dict, **self.default_resource_consumption_dict})
                #action = Action(origin_node, destination_node, cost, contribution_vector, min_resource_dict, resource_consumption_dict, max_resource_dict)
                
 
                _,min_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_min_resource_dict)     
                _,resource_consumption_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_resource_consumption_dict)     
                _,max_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_max_resource_dict)     
                indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,partial_max_resource_dict)
                
                ##print('trans_min_input')
                #print(trans_min_input)
                #print('min_resource_vec')
                #print(min_resource_vec)
                #print('resource_consumption_vec')
                #print(resource_consumption_vec)
                #print('max_resource_vec')
                #print(max_resource_vec)
                #input('')
                
                #remember when applying the max_resource_vec we only apply it over indices_non_zero_max
 
                #action = Action(origin_node, destination_node, cost, contribution_vector, trans_min_input, trans_term_add, trans_term_min,min_resource_vec,resource_consumption_vec,indices_non_zero_max,max_resource_vec)
 
                action = Action(trans_min_input,trans_term_add,trans_term_min,destination_node,origin_node,contribution_vector,cost,min_resource_vec,resource_consumption_vec,indices_apply_min_to,max_resource_vec)
                self.actions[origin_node, destination_node] = [action]
                #start delete
                print(f'origin:{origin_node},destination:{destination_node}')
                print(f'trans_min_input:{trans_min_input},trans_term_min:{trans_term_min},trans_term_add:{trans_term_add}')
                print(f'trans_term_min:{min_resource_vec.toarray()},trans_term_vec{resource_consumption_vec.toarray()},indices_apply_min_to{indices_apply_min_to},max_resource_vec{max_resource_vec.toarray()}')
                print('checkhere')
        print('checkhere')
                
 
    def _create_null_action_info(self):
        trans_min_input = {}
        trans_term_add = {}
        trans_term_min = {}
        for res_name in self.resource_name_to_index.keys():
            trans_min_input[res_name] = 0
            trans_term_add[res_name] = 0
            trans_term_min[res_name] = np.inf
        contribution_vector = np.zeros(len(self.rhs_vector))
        cost = 0
        min_resource_vec = np.zeros(self.number_of_resources)
        resource_consumption_vec = np.zeros(self.number_of_resources)
        indices_non_zero_max = []    
        max_resource_vec = np.full(self.number_of_resources, np.inf)
        self.initial_null_actions['trans_min_input'] = trans_min_input
        self.initial_null_actions['trans_term_add'] = trans_term_add
        self.initial_null_actions['trans_term_min'] = trans_term_min
        self.initial_null_actions['contribution_vector'] = contribution_vector
        self.initial_null_actions['cost'] = cost
        self.initial_null_actions['min_resource_vec'] = min_resource_vec
        self.initial_null_actions['resource_consumption_vec'] = resource_consumption_vec
        self.initial_null_actions['indices_non_zero_max'] = indices_non_zero_max
        self.initial_null_actions['max_resource_vec'] = max_resource_vec
 
    def _create_initial_res_actions(self):
        """Note to Julian: No code exists for this yet."""
 
        for node in self.nodes:
            if node > 0:
                #one for the source to each customer
                self.initial_res_actions.update(self.actions[-1, node])
                #one each cusotmer to the sink
                self.initial_res_actions.update(self.actions[node, -2])
 
    def _create_initial_res_states(self):
        """Note to Julian: No code exists for this yet."""
        full_resource_dict = np.ones(self.number_of_resources)
        full_resource_dict[0] = self.capacity
        full_resource_vec = csr_matrix(full_resource_dict.reshape(1, -1))
        self.initial_resource_vector = full_resource_vec
        empty_resource_dict = np.zeros(self.number_of_resources)
        empty_resource_vec = csr_matrix(empty_resource_dict.reshape(1, -1))
 
        #one for the source
        source_state = State(-1,full_resource_vec,0,True,False)
        self.initial_res_states.add(source_state)
 
        #one for the sink
        sink_state = State(-2,empty_resource_vec,0,False,True)
 
        self.initial_res_states.add(sink_state)
 
        #one for each node with capacity remaining at maximum
        for node in self.nodes:
            if node > 0:
                this_res = np.zeros(self.number_of_resources)
                this_res[node] =1
                this_res[0]=self.capacity
                node_state = State(node,csr_matrix(this_res.reshape(1,-1)),0,False,False)
                self.initial_res_states.add(node_state)
        print('check here')
        
    def _sorted_nearest_node(self):
        #self.cost_matrix =  {(u,v): self.distance(u,v) for u in self.nodes for v in self.nodes }
        self.neighbors_by_distance = {
            u: sorted(
                [v for v in self.nodes if v != u],
                key=lambda v: self.distance(u, v)
            )
            for u in self.nodes
        }
    
    def _define_state_update_module(self):
        """This is where we define how res_states is updated after pricing. We are using the `standard_CVRP` module for this definition."""
        self.state_update_module = CVRP_state_update_function(self.nodes, self.actions,self.capacity, self.demands, self.neighbors_by_distance, self.initial_resource_vector,self.resource_name_to_index,self.number_of_resources)
    
    def _generate_neighbors(self):
        pass
 
    def _distance(self, origin, destination):
        """Helper function to get costs of actions easily."""
        x1, y1 = self.coordinates[origin]
        x2, y2 = self.coordinates[destination]
 
        return hypot(x2 - x1, y2 - y1)
 