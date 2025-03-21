from src.problems.optimization_problem import OptimizationProblem
from src.common.action import Action
from src.common.state import State
from src.common.helper import Helper
from src.algorithm.update_states.LoadAI_state_generation_input import LoadAI_state_input
from typing import List, Dict
from numpy import zeros, ones, append
from scipy.sparse import csr_matrix
from collections import ChainMap
from math import hypot, radians, sin, cos, sqrt, asin
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
# CONSTANTS
VOLUME_CAPACITY = 3000
WEIGHT_CAPACITY = 45000
MAX_COMBINED_LOADS = 3
HOS_DRIVE_TIME = 11 * 60
HOS_WORK_TIME = 14 * 60
HOS_REST_TIME = 9 * 60
AVERAGE_SPEED = 55 / 60
MIN_DISTANCE_SAVING = 100
STANDARD_SERVICE_TIME = 2 * 60

class loadAI(OptimizationProblem):
    def __init__(self, problem_instance_file_name, file_type: str = "Standard_Form"):
        
        super().__init__(problem_instance_file_name, file_type)
        
    def solve(self):
        return super().solve()
    
    def _load_data_from_file(self):
        """Parses data from file. Handles different types of files depending on `file_type` property."""
        match self.file_type:
            case "Standard_Form":
                self._load_standard_form_file()

            # Call file parsing functions for other file types here
            case _:
                raise Exception(f"No file parsing logic available for file type {self.file_type}")
        
    def _load_standard_form_file(self):
        """Loads raw data for `Standard_Form` file type. This function has not been reviewed and probably has many bugs right now."""
        self.volume_capacity: int = VOLUME_CAPACITY
        self.weight_capacity: int = WEIGHT_CAPACITY
        self.hos_drive_time: int = HOS_DRIVE_TIME
        self.hos_work_time: int = HOS_WORK_TIME
        self.max_combined_loads: int = MAX_COMBINED_LOADS
        self.maximum_time: int = None
        self.weight_demands : Dict[int, float] = {}
        self.volume_demands : Dict[int, float] = {}
        self.coordinates : Dict[int, float] = {}
        self.time_window_start : Dict[int, float] = {}
        self.time_window_end : Dict[int, float] = {}
        self.service_time : Dict[int, float] = {}
        self.pickup_to_dropoff : Dict[int, int] = {}
        self.dropoff_to_pickup : Dict[int, int] = {}
        self.problem_info = {}
        self.problem_info['volume_capacity'] = VOLUME_CAPACITY
        self.problem_info['weight_capacity'] = WEIGHT_CAPACITY
        self.problem_info['hos_drive_time'] = HOS_DRIVE_TIME
        self.problem_info['hos_work_time'] = HOS_WORK_TIME
        self.problem_info['max_combined_loads'] = MAX_COMBINED_LOADS
        file_path = self.problem_instance_file_name # FIX THIS FOR THE PROPER FILE PATH
        df = pd.read_csv(file_path)
        self._create_null_action_info()

        # Convert time columns to datetime objects
        df["Pickup Appointment Start Date Time"] = pd.to_datetime(df["Pickup Appointment Start Date Time"])
        df["Pickup Appointment End Date Time"] = pd.to_datetime(df["Pickup Appointment End Date Time"])
        df["Delivery Appointment Start Date Time"] = pd.to_datetime(df["Delivery Appointment Start Date Time"])
        df["Delivery Appointment End Date Time"] = pd.to_datetime(df["Delivery Appointment End Date Time"])

        # Find the maximum time (latest time in dataset)
        latest_time = max(df["Delivery Appointment End Date Time"].max(), df["Pickup Appointment End Date Time"].max())
        earliest_time = min(df["Pickup Appointment Start Date Time"].min(), df["Delivery Appointment Start Date Time"].min())
        self.maximum_time = int((latest_time - earliest_time).total_seconds() // 60)  # Convert to minutes

        num_customers = len(df)
        
        for index, row in df.iterrows():
            pickup_id = index + 1
            dropoff_id = pickup_id + num_customers
            
            # Assign weight and volume demands
            self.weight_demands[pickup_id] = row["Total Weight"]
            self.volume_demands[pickup_id] = row["Pallet Count"]
            # self.weight_demands[dropoff_id] = 0  # Dropoffs have no demand
            # self.volume_demands[dropoff_id] = 0
            
            # Store coordinates
            self.coordinates[pickup_id] = (row["Pickup Lat"], row["Pickup Lon"])
            self.coordinates[dropoff_id] = (row["Delivery Lat"], row["Delivery Lon"])
            
            # Normalize time windows
            pickup_start = int((row["Pickup Appointment Start Date Time"] - earliest_time).total_seconds() // 60)
            pickup_end = int((row["Pickup Appointment End Date Time"] - earliest_time).total_seconds() // 60)
            delivery_start = int((row["Delivery Appointment Start Date Time"] - earliest_time).total_seconds() // 60)
            delivery_end = int((row["Delivery Appointment End Date Time"] - earliest_time).total_seconds() // 60)
            
            self.time_window_start[pickup_id] = self.maximum_time - pickup_start
            self.time_window_end[pickup_id] = self.maximum_time - pickup_end
            self.time_window_start[dropoff_id] = self.maximum_time - delivery_start
            self.time_window_end[dropoff_id] = self.maximum_time - delivery_end
            
            # Service time (assumed to be 0 for now, but can be updated if needed)
            self.service_time[pickup_id] = STANDARD_SERVICE_TIME
            self.service_time[dropoff_id] = STANDARD_SERVICE_TIME
            
            # Assign pickup-dropoff relationships
            self.pickup_to_dropoff[pickup_id] = dropoff_id
            self.dropoff_to_pickup[dropoff_id] = pickup_id
        self.time_window_start[-1] = np.inf
        self.time_window_end[-1] = 0
        self.time_window_start[-2] = np.inf
        self.time_window_end[-2] = 0
    def _build_problem_model(self):
        # NODES
        self.nodes.append(-1)
        self.number_of_customers = len(self.pickup_to_dropoff)
        for node in self.weight_demands:
            self.nodes.append(node)
        
        for node in self.weight_demands:
            self.nodes.append(round(node + self.number_of_customers))
        
        for pickup_node in self.pickup_to_dropoff:
            skip_node = round(round(pickup_node + (2 * self.number_of_customers)))
            self.nodes.append(skip_node)
        # for n1, n2 in self.pickup_to_dropoff.items():
        #     self.nodes.append(n1)
        #     self.nodes.append(n2)
        self.nodes.append(-2)
        #print('self.nodes')
        #print(self.nodes)
        #input('---')
        # EXOG RHS
        self.rhs_vector = ones(self.number_of_customers)
        idx = 0
        for pickup_node in self.pickup_to_dropoff:
            self.rhs_dict[str(("Cover", pickup_node))] = 1
            self.rhs_constraint_name_to_index[str(("Cover", pickup_node))] = idx
            self.rhs_index_to_constraint_name[idx] = str(("Cover", pickup_node))
            idx += 1

        # INITIAL RESOURCE STATE
        self._populate_initial_resources()

        # ACTIONS
        self._create_default_resource_values()
        self._create_source_sink_actions()
        self._create_pickup_to_pickup_actions()
        self._create_pickup_to_dropoff_actions()
        self._create_dropoff_to_pickup_actions()
        self._create_dropoff_to_dropoff_actions()
        self._create_skip_actions()

        



    def _populate_initial_resources(self):
        """Helper function to handle building the resource dicts/vector"""
        idx = 0

        self.initial_resource_dict["weight"] = self.weight_capacity
        self.initial_resource_vector = append(self.initial_resource_vector, self.weight_capacity)
        self.resource_name_to_index["weight"] = idx
        self.resource_index_to_name[idx] = "weight"
        idx += 1

        self.initial_resource_dict["volume"] = self.volume_capacity
        self.initial_resource_vector = append(self.initial_resource_vector, self.volume_capacity)
        self.resource_name_to_index["volume"] = idx
        self.resource_index_to_name[idx] = "volume"
        idx += 1

        self.initial_resource_dict["time"] = self.maximum_time
        self.initial_resource_vector = append(self.initial_resource_vector, self.maximum_time)
        self.resource_name_to_index["time"] = idx
        self.resource_index_to_name[idx] = "time"
        idx += 1

        self.initial_resource_dict["max_combined_loads"] = self.max_combined_loads
        self.initial_resource_vector = append(self.initial_resource_vector, self.max_combined_loads)
        self.resource_name_to_index["max_combined_loads"] = idx
        self.resource_index_to_name[idx] = "max_combined_loads"
        idx += 1

        # self.initial_resource_dict["HOS_drive_time"] = HOS_DRIVE_TIME
        # self.initial_resource_vector = append(self.initial_resource_vector, HOS_DRIVE_TIME)
        # self.resource_name_to_index["HOS_drive_time"] = idx
        # self.resource_index_to_name[idx] = "HOS_drive_time"
        # idx += 1

        # self.initial_resource_dict["HOS_work_time"] = HOS_WORK_TIME
        # self.initial_resource_vector = append(self.initial_resource_vector, HOS_WORK_TIME)
        # self.resource_name_to_index["HOS_work_time"] = idx
        # self.resource_index_to_name[idx] = "HOS_work_time"
        # idx += 1

        for pickup_node in self.pickup_to_dropoff:
            self.initial_resource_dict[str(("may_pickup", pickup_node))] = 1
            self.initial_resource_vector = append(self.initial_resource_vector, 1)
            self.resource_name_to_index[str(("may_pickup", pickup_node))] = idx
            self.resource_index_to_name[idx] = str(("may_pickup", pickup_node))
            idx += 1

        for dropoff_node in self.dropoff_to_pickup:
            self.initial_resource_dict[str(("may_avoid_dropoff", dropoff_node))] = 1
            self.initial_resource_vector = append(self.initial_resource_vector, 1)
            self.resource_name_to_index[str(("may_avoid_dropoff", dropoff_node))] = idx
            self.resource_index_to_name[idx] = str(("may_avoid_dropoff", dropoff_node))
            idx += 1

        self.number_of_resources = len(self.initial_resource_dict)
        self.initial_resource_vector=csr_matrix(self.initial_resource_vector.reshape(1, -1))

    def _empty_resource_vec(self) -> csr_matrix:
        return csr_matrix(self.empty_resource_array.reshape(1, -1))
    
    def _full_resource_vec(self):
        return csr_matrix(self.full_resource_array.reshape(1, -1))


    def _create_default_resource_values(self):
        """Defines default resource values for actions."""
        self.empty_resource_array = zeros(self.number_of_resources)

        self.default_trans_min_input = {
            "volume": 0,
            "weight": 0,
            "time": 0,
            "max_combined_loads": 0,
        }
        self.default_trans_term_vec = {
            "volume": 0,
            "weight": 0,
            "time": 0,
            "max_combined_loads": 0,
        }
        self.default_trans_term_min = {
            "volume": self.volume_capacity,
            "weight": self.weight_capacity,
            "time": self.maximum_time,
            "max_combined_loads": self.max_combined_loads,
        }

        for pickup in self.pickup_to_dropoff:
            self.default_trans_min_input[str(("may_pickup", pickup))] = 0
            self.default_trans_term_vec[str(("may_pickup", pickup))] = 0
            self.default_trans_term_min[str(("may_pickup", pickup))] = 1

        for dropoff in self.dropoff_to_pickup:
            self.default_trans_min_input[str(("may_avoid_dropoff", dropoff))] = 0
            self.default_trans_term_vec[str(("may_avoid_dropoff", dropoff))] = 0
            self.default_trans_term_min[str(("may_avoid_dropoff", dropoff))] = 1

        self.full_resource_array = zeros(len(self.resource_name_to_index))

        for resource_name, index in self.resource_name_to_index.items():
            if resource_name in self.default_trans_term_min:
                self.full_resource_array[index] = self.default_trans_term_min[resource_name]


    def _create_source_sink_actions(self):
        for destination_node in self.pickup_to_dropoff:
            origin_node = -1  # Source
            cost = 0
            exog_contrib_vec = self._default_contribution_vector()
            partial_trans_min_input = {}
            partial_trans_term_vec = {}
            partial_trans_term_min = {"time": self.time_window_start[destination_node]}
            trans_min_input = ChainMap(partial_trans_min_input, self.default_trans_min_input)
            trans_term_vec = ChainMap(partial_trans_term_vec, self.default_trans_term_vec)
            trans_term_min = ChainMap(partial_trans_term_min, self.default_trans_term_min)
            _,min_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_min_input)     
            _,resource_consumption_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_term_vec)     
            _,max_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,trans_term_min)     
            indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,trans_term_min)

            action = Action(trans_min_input, trans_term_vec, trans_term_min, destination_node, origin_node, exog_contrib_vec, cost, min_resource_vec, resource_consumption_vec, indices_apply_min_to, max_resource_vec, self._full_resource_vec(), self._empty_resource_vec())
            self.actions[origin_node, destination_node] = [action]

        for origin_node in self.dropoff_to_pickup:
            origin_node = origin_node
            destination_node = -2  # Sink
            cost = 0
            exog_contrib_vec = self._default_contribution_vector()
            partial_trans_min_input = {}
            partial_trans_term_vec = {}
            partial_trans_term_min = {}
            for dropoff_node in self.dropoff_to_pickup:
                if dropoff_node == origin_node:
                    continue
                partial_trans_min_input[str(("may_avoid_dropoff", dropoff_node))] = 1
                partial_trans_term_vec[str(("may_avoid_dropoff", dropoff_node))] = -1

            trans_min_input = ChainMap(partial_trans_min_input, self.default_trans_min_input)
            trans_term_vec = ChainMap(partial_trans_term_vec, self.default_trans_term_vec)
            trans_term_min = ChainMap(partial_trans_term_min, self.default_trans_term_min)

            _,min_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_min_input)     
            _,resource_consumption_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_term_vec)     
            _,max_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_term_min)     
            indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,partial_trans_term_min)

            action = Action(trans_min_input, trans_term_vec, trans_term_min, destination_node, origin_node, exog_contrib_vec, cost, min_resource_vec, resource_consumption_vec, indices_apply_min_to, max_resource_vec, self._full_resource_vec(), self._empty_resource_vec())
            self.actions[origin_node, destination_node] = [action]
            
        
    def _create_pickup_to_pickup_actions(self):
        for origin_node in self.pickup_to_dropoff:
            for destination_node in self.pickup_to_dropoff:
                if origin_node == destination_node:
                    continue

                cost = self._haversine_distance(origin_node, destination_node)
                exog_contrib_vec = self._default_contribution_vector()
                cover_constraint_index = self.rhs_constraint_name_to_index[str(("Cover", origin_node))]
                exog_contrib_vec[cover_constraint_index] = 1
                partial_trans_min_input = {"time": self._travel_time(origin_node, destination_node) + self.service_time[origin_node] + self.time_window_end[destination_node], 
                                           "volume": self.volume_demands[origin_node] + self.volume_demands[destination_node],
                                           "weight": self.weight_demands[origin_node] + self.weight_demands[destination_node],
                                           "max_combined_loads": 1,
                                           str(("may_pickup", destination_node)): 1}
                partial_trans_term_vec = {"time": -self._travel_time(origin_node, destination_node) - self.service_time[origin_node], 
                                           "volume": -self.volume_demands[origin_node],
                                           "weight": -self.weight_demands[origin_node],
                                           "max_combined_loads": -1,
                                           str(("may_pickup", origin_node)): -1,
                                           str(("may_avoid_dropoff", self.pickup_to_dropoff[origin_node])): -1}
                partial_trans_term_min = {"time": self.time_window_start[destination_node]}
                trans_min_input = ChainMap(partial_trans_min_input, self.default_trans_min_input)
                trans_term_vec = ChainMap(partial_trans_term_vec, self.default_trans_term_vec)
                trans_term_min = ChainMap(partial_trans_term_min, self.default_trans_term_min)
                
                _,min_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_min_input)     
                _,resource_consumption_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_term_vec)     
                _,max_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_term_min)     
                indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,partial_trans_term_min)

                action = Action(trans_min_input, trans_term_vec, trans_term_min, destination_node, origin_node, exog_contrib_vec, cost, min_resource_vec, resource_consumption_vec, indices_apply_min_to, max_resource_vec, self._full_resource_vec(), self._empty_resource_vec())
                self.actions[origin_node, destination_node] = [action]
        
    def _create_pickup_to_dropoff_actions(self):
        for origin_node in self.pickup_to_dropoff:
            for destination_node in self.dropoff_to_pickup:
                cost = self._haversine_distance(origin_node, destination_node)
                exog_contrib_vec = self._default_contribution_vector()
                cover_constraint_index = self.rhs_constraint_name_to_index[str(("Cover", origin_node))]
                exog_contrib_vec[cover_constraint_index] = 1
                partial_trans_min_input = {"time": self._travel_time(origin_node, destination_node) + self.service_time[origin_node] + self.time_window_end[destination_node]}
                partial_trans_term_vec = {"time": -self._travel_time(origin_node, destination_node) - self.service_time[origin_node], 
                                           "volume": -self.volume_demands[origin_node],
                                           "weight": -self.weight_demands[origin_node],
                                           "max_combined_loads": -1,
                                           str(("may_pickup", origin_node)): -1,
                                           str(("may_avoid_dropoff", self.pickup_to_dropoff[origin_node])): -1}
                partial_trans_term_min = {"time": self.time_window_start[destination_node]}
                trans_min_input = ChainMap(partial_trans_min_input, self.default_trans_min_input)
                trans_term_vec = ChainMap(partial_trans_term_vec, self.default_trans_term_vec)
                trans_term_min = ChainMap(partial_trans_term_min, self.default_trans_term_min)
                
                _,min_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_min_input)     
                _,resource_consumption_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_term_vec)     
                _,max_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_term_min)     
                indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,partial_trans_term_min)

                action = Action(trans_min_input, trans_term_vec, trans_term_min, destination_node, origin_node, exog_contrib_vec, cost, min_resource_vec, resource_consumption_vec, indices_apply_min_to, max_resource_vec, self._full_resource_vec(), self._empty_resource_vec())
                self.actions[origin_node, destination_node] = [action]
            
        
    def _create_dropoff_to_pickup_actions(self):
        for origin_node in self.dropoff_to_pickup:
            for destination_node in self.pickup_to_dropoff:
                origin_pickup_node = self.dropoff_to_pickup[origin_node]
                if origin_pickup_node == destination_node:
                    continue
                cost = self._haversine_distance(origin_node, destination_node)
                exog_contrib_vec = self._default_contribution_vector()
                partial_trans_min_input = {"time": self._travel_time(origin_node, destination_node) + self.service_time[origin_node] + self.time_window_end[destination_node], 
                                           "volume": self.volume_demands[destination_node] - self.volume_demands[origin_pickup_node],
                                           "weight": self.weight_demands[destination_node] - self.weight_demands[origin_pickup_node],
                                           "max_combined_loads": 1,
                                           str(("may_pickup", destination_node)): 1}
                partial_trans_term_vec = {"time": -self._travel_time(origin_node, destination_node) - self.service_time[origin_node], 
                                           "volume": -self.volume_demands[origin_pickup_node],
                                           "weight": -self.weight_demands[origin_pickup_node],
                                           str(("may_avoid_dropoff", origin_node)): 1}
                partial_trans_term_min = {"time": self.time_window_start[destination_node]}
                trans_min_input = ChainMap(partial_trans_min_input, self.default_trans_min_input)
                trans_term_vec = ChainMap(partial_trans_term_vec, self.default_trans_term_vec)
                trans_term_min = ChainMap(partial_trans_term_min, self.default_trans_term_min)
                
                _,min_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_min_input)     
                _,resource_consumption_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_term_vec)     
                _,max_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_term_min)     
                indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,partial_trans_term_min)

                action = Action(trans_min_input, trans_term_vec, trans_term_min, destination_node, origin_node, exog_contrib_vec, cost, min_resource_vec, resource_consumption_vec, indices_apply_min_to, max_resource_vec, self._full_resource_vec(), self._empty_resource_vec())
                self.actions[origin_node, destination_node] = [action]
        
        
    def _create_dropoff_to_dropoff_actions(self):
        for origin_node in self.dropoff_to_pickup:
            for destination_node in self.dropoff_to_pickup:
                if origin_node == destination_node:
                    continue
                origin_pickup_node = self.dropoff_to_pickup[origin_node]
                cost = self._haversine_distance(origin_node, destination_node)
                exog_contrib_vec = self._default_contribution_vector()
                partial_trans_min_input = {"time": self._travel_time(origin_node, destination_node) + self.service_time[origin_node] + self.time_window_end[destination_node]}
                partial_trans_term_vec = {"time": -self._travel_time(origin_node, destination_node) - self.service_time[origin_node], 
                                           "volume": -self.volume_demands[origin_pickup_node],
                                           "weight": -self.weight_demands[origin_pickup_node],
                                           str(("may_avoid_dropoff", origin_node)): 1}
                partial_trans_term_min = {"time": self.time_window_start[destination_node]}
                trans_min_input = ChainMap(partial_trans_min_input, self.default_trans_min_input)
                trans_term_vec = ChainMap(partial_trans_term_vec, self.default_trans_term_vec)
                trans_term_min = ChainMap(partial_trans_term_min, self.default_trans_term_min)
                
                _,min_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_min_input)     
                _,resource_consumption_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_term_vec)     
                _,max_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_term_min)     
                indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,partial_trans_term_min)

                action = Action(trans_min_input, trans_term_vec, trans_term_min, destination_node, origin_node, exog_contrib_vec, cost, min_resource_vec, resource_consumption_vec, indices_apply_min_to, max_resource_vec, self._full_resource_vec(), self._empty_resource_vec())
                self.actions[origin_node, destination_node] = [action]
        
        
    def _create_skip_actions(self):
        for destination_node in self.pickup_to_dropoff:
            origin_node = -1
            cost = self._slack(destination_node)
            exog_contrib_vec = self._default_contribution_vector()
            cover_constraint_index = self.rhs_constraint_name_to_index[str(("Cover", destination_node))]
            exog_contrib_vec[cover_constraint_index] = 1

            trans_min_input = ChainMap({}, self.default_trans_min_input)
            trans_term_vec = ChainMap({}, self.default_trans_term_vec)
            trans_term_min = ChainMap({}, self.default_trans_term_min)
            
            _,min_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,{})     
            _,resource_consumption_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,{})     
            _,max_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,{})     
            indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,{})

            destination_node = destination_node + 2 * self.number_of_customers
            action = Action(trans_min_input, trans_term_vec, trans_term_min, destination_node, origin_node, exog_contrib_vec, cost, min_resource_vec, resource_consumption_vec, indices_apply_min_to, max_resource_vec, self._full_resource_vec(), self._empty_resource_vec())
            self.actions[origin_node, destination_node] = [action]
               
        for origin_node in self.pickup_to_dropoff:
            destination_node = -2
            cost = 0
            exog_contrib_vec = self._default_contribution_vector()

            trans_min_input = ChainMap({}, self.default_trans_min_input)
            trans_term_vec = ChainMap({}, self.default_trans_term_vec)
            trans_term_min = ChainMap({}, self.default_trans_term_min)
            
            _,min_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,{})     
            _,resource_consumption_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,{})     
            _,max_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,{})     
            indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,{})

            origin_node = origin_node + 2 * self.number_of_customers
            action = Action(trans_min_input, trans_term_vec, trans_term_min, destination_node, origin_node, exog_contrib_vec, cost, min_resource_vec, resource_consumption_vec, indices_apply_min_to, max_resource_vec, self._full_resource_vec(), self._empty_resource_vec())
            self.actions[origin_node, destination_node] = [action]
    

    def _default_contribution_vector(self):
        return zeros(self.number_of_customers)
        
    def _distance(self, origin, destination):
        x1, y1 = self.coordinates[origin]
        x2, y2 = self.coordinates[destination]
        return hypot(x2 - x1, y2 - y1)
    
    def _haversine_distance(self, origin, destination):
        EARTH_RADIUS = 3958.8  # Radius of Earth in miles
        lat1, lon1 = map(radians, self.coordinates[origin])
        lat2, lon2 = map(radians, self.coordinates[destination])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))

        return EARTH_RADIUS * c  # Distance in miles

    def _travel_time(self, origin, destination):
        distance = self._haversine_distance(origin, destination)
        drive_time = distance / AVERAGE_SPEED
        number_of_rests = int(drive_time / HOS_DRIVE_TIME)
        travel_time = drive_time + number_of_rests * HOS_REST_TIME
        return travel_time
    
    def _slack(self, pickup_node):
        dropoff_node = self.pickup_to_dropoff[pickup_node]
        pickup_dropoff_direct_distance = self._haversine_distance(pickup_node, dropoff_node)
        slack_coeff = max(0,pickup_dropoff_direct_distance - MIN_DISTANCE_SAVING)
        if slack_coeff > 0:
            return slack_coeff
        return 0
        

    def _create_initial_res_actions(self):
        for skip_node in self.nodes[((self.number_of_customers * 2) + 1):-1]:
            self.initial_res_actions.add(self.actions[-1, skip_node][0])
            self.initial_res_actions.add(self.actions[skip_node, -2][0])

    
    def _create_initial_res_states(self):
        self.initial_res_states.add(State(-1, self._full_resource_vec(), 0, True, False))

        for skip_node in self.nodes[(self.number_of_customers * 2)+1:]:
            self.initial_res_states.add(State(skip_node, self._full_resource_vec(), 0, False, False))

        self.initial_res_states.add(State(-2, self._empty_resource_vec(), 0, False, True))
        
    
    def _create_null_action_info(self):
        full_resource_array = np.ones(self.number_of_resources)
        full_resource_vec = csr_matrix(full_resource_array.reshape(1, -1))
        empty_resource_array = np.zeros(self.number_of_resources)
        empty_resource_vec = csr_matrix(empty_resource_array.reshape(1, -1))
        trans_min_input = {}
        trans_term_add = {}
        trans_term_min = {}
        for res_name in self.resource_name_to_index.keys():
            trans_min_input[res_name] = 0
            trans_term_add[res_name] = 0
            trans_term_min[res_name] = np.inf
        #contribution_vector = np.zeros(len(self.rhs_vector))
        contribution_vector = np.zeros(len(self.rhs_vector))
        cost = 0
        min_resource_vec = np.zeros(self.number_of_resources)
        resource_consumption_vec = np.zeros(self.number_of_resources)
        indices_apply_min_to = []    
        max_resource_vec = np.full(self.number_of_resources, np.inf)
       # self.initial_null_actions['trans_min_input'] = trans_min_input
       # self.initial_null_actions['trans_term_add'] = trans_term_add
       # self.initial_null_actions['trans_term_min'] = trans_term_min
       # self.initial_null_actions['contribution_vector'] = contribution_vector
       # self.initial_null_actions['cost'] = cost
       # self.initial_null_actions['min_resource_vec'] = min_resource_vec
       # self.initial_null_actions['resource_consumption_vec'] = resource_consumption_vec
       # self.initial_null_actions['indices_non_zero_max'] = indices_non_zero_max
       # self.initial_null_actions['max_resource_vec'] = max_resource_vec

        self.the_single_null_action= Action(trans_min_input,trans_term_add,trans_term_min,None,None,contribution_vector,cost,min_resource_vec,resource_consumption_vec,indices_apply_min_to,max_resource_vec,full_resource_vec,empty_resource_vec)
    def _create_travel_time(self):
        self.travel_time = {}
        nodes = []
        for n1,n2 in self.pickup_to_dropoff.items():
            nodes.append(n1)
            nodes.append(n2)
        for n1 in nodes:
            for n2 in nodes:
                if n1 != n2:
                    self.travel_time[(n1,n2)] = self._travel_time(n1,n2)
                else:
                    self.travel_time[(n1,n2)] =0
        nodes.append(-1)
        nodes.append(-2)
        return nodes
    def _create_nearest_node(self,nodes,k):
        neighbors_by_distance = {
            u: sorted(
                [v for v in nodes if v != u and v not in {-1, -2}],
                key=lambda v: self.travel_time[(u, v)]
            )
            for u in nodes if u not in {-1, -2}
        }
        k = min(k,len(self.nodes)-1)
        neighbors = {}
        for u, nodes in neighbors_by_distance.items():
            neighbors[u] = nodes[:k]
            if u == -1:
                neighbors[u].remove(-2)
            if u == -2:
                neighbors[u].remove(-1)
        return neighbors_by_distance, neighbors
    def _define_state_update_module(self):
        # ASSIGN STATE UPDATE MODULE HERE
        nodes = self._create_travel_time()
        neighbors_by_distance, neighbors = self._create_nearest_node(nodes,10)
        self.plot_pickup_dropoff_locations()
        self.state_update_module = LoadAI_state_input(self.nodes, self.actions, self.weight_capacity, self.weight_demands, self.time_window_start, self.time_window_end, self.pickup_to_dropoff, self.dropoff_to_pickup, neighbors_by_distance, neighbors, self.travel_time, self.initial_resource_vector, self.resource_name_to_index, self.number_of_resources, self.problem_info)
    def plot_pickup_dropoff_locations(self):
        """
        Create a visualization of pickup and dropoff locations using different colors.
        
        Parameters:
        - self: The class instance containing the required attributes
        """
        # Create a new figure
        plt.figure(figsize=(12, 10))
        
        # Extract coordinates for pickup nodes
        pickup_nodes = list(self.pickup_to_dropoff.keys())
        pickup_lats = [self.coordinates[node][0] for node in pickup_nodes]
        pickup_longs = [self.coordinates[node][1] for node in pickup_nodes]
        
        # Extract coordinates for dropoff nodes
        dropoff_nodes = list(self.dropoff_to_pickup.keys())
        dropoff_lats = [self.coordinates[node][0] for node in dropoff_nodes]
        dropoff_longs = [self.coordinates[node][1] for node in dropoff_nodes]
        
        # Plot the points with different colors
        plt.scatter(pickup_longs, pickup_lats, c='green', marker='o', s=100, 
                    label='Pickup Nodes', alpha=0.8, edgecolors='darkgreen')
        plt.scatter(dropoff_longs, dropoff_lats, c='red', marker='s', s=100, 
                    label='Dropoff Nodes', alpha=0.8, edgecolors='darkred')
        
        # Add node labels
        for node in pickup_nodes:
            plt.annotate(str(node), (self.coordinates[node][1], self.coordinates[node][0]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
            
        for node in dropoff_nodes:
            plt.annotate(str(node), (self.coordinates[node][1], self.coordinates[node][0]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add title and labels
        plt.title('Pickup and Dropoff Locations', fontsize=16)
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        
        # Add a legend
        plt.legend(fontsize=12)
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Improve layout
        plt.tight_layout()
        
        # Show the plot
        #plt.savefig('pickup_dropoff_map.png', dpi=300, bbox_inches='tight')
        #plt.show()
        
        return plt
