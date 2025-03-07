from src.problems.optimization_problem import OptimizationProblem
from src.common.action import Action
from typing import List, Dict
from numpy import zeros, ones, append
from collections import ChainMap
from math import hypot
import pandas as pd

# CONSTANTS
VOLUME_CAPACITY = 3000
WEIGHT_CAPACITY = 45000
MAX_COMBINED_LOADS = 3
HOS_DRIVE_TIME = 11 * 60
HOS_WORK_TIME = 14 * 60
HOS_REST_TIME = 9 * 60
AVERAGE_SPEED = 55 / 60
MIN_DISTANCE_SAVING = 100

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
        
        file_path = self.problem_instance_file_name # FIX THIS FOR THE PROPER FILE PATH
        df = pd.read_csv(file_path)

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
            self.weight_demands[dropoff_id] = 0  # Dropoffs have no demand
            self.volume_demands[dropoff_id] = 0
            
            # Store coordinates
            self.coordinates[pickup_id] = (row["Pickup Lat"], row["Pickup Lon"])
            self.coordinates[dropoff_id] = (row["Delivery Lat"], row["Delivery Lon"])
            
            # Normalize time windows
            pickup_start = int((row["Pickup Appointment Start Date Time"] - earliest_time).total_seconds() // 60)
            pickup_end = int((row["Pickup Appointment End Date Time"] - earliest_time).total_seconds() // 60)
            delivery_start = int((row["Delivery Appointment Start Date Time"] - earliest_time).total_seconds() // 60)
            delivery_end = int((row["Delivery Appointment End Date Time"] - earliest_time).total_seconds() // 60)
            
            self.time_window_start[pickup_id] = pickup_start
            self.time_window_end[pickup_id] = pickup_end
            self.time_window_start[dropoff_id] = delivery_start
            self.time_window_end[dropoff_id] = delivery_end
            
            # Service time (assumed to be 0 for now, but can be updated if needed)
            self.service_time[pickup_id] = 0
            self.service_time[dropoff_id] = 0
            
            # Assign pickup-dropoff relationships
            self.pickup_to_dropoff[pickup_id] = dropoff_id
            self.dropoff_to_pickup[dropoff_id] = pickup_id

    def _build_problem_model(self):
        # NODES
        self.nodes.append[-1]
        self.number_of_customers = len(self.pickup_to_dropoff)
        for node in self.weight_demands:
            self.nodes.append(node)

        for node in self.weight_demands:
            self.nodes.append(round(node + self.number_of_customers))
        
        for pickup_node in self.pickup_to_dropoff:
            skip_node = round(round(pickup_node + (2 * self.number_of_customers)))
            self.nodes.append(skip_node)

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

        self.initial_resource_dict["weight_remain"] = self.weight_capacity
        self.initial_resource_vector = append(self.initial_resource_vector, self.weight_capacity)
        self.resource_name_to_index["weight_remain"] = idx
        self.resource_index_to_name[idx] = "weight_remain"
        idx += 1

        self.initial_resource_dict["volume_remain"] = self.volume_capacity
        self.initial_resource_vector = append(self.initial_resource_vector, self.volume_capacity)
        self.resource_name_to_index["volume_remain"] = idx
        self.resource_index_to_name[idx] = "volume_remain"
        idx += 1

        self.initial_resource_dict["time_remain"] = self.maximum_time
        self.initial_resource_vector = append(self.initial_resource_vector, self.maximum_time)
        self.resource_name_to_index["time_remain"] = idx
        self.resource_index_to_name[idx] = "time_remain"
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


    def _create_default_resource_values(self):
        """Defines default resource values for actions."""
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
            self.default_trans_min_input[("may_pickup", pickup)] = 0
            self.default_trans_term_vec[("may_pickup", pickup)] = 0
            self.default_trans_term_min[("may_pickup", pickup)] = 1

        for dropoff in self.dropoff_to_pickup:
            self.default_trans_min_input[("may_avoid_dropoff", dropoff)] = 0
            self.default_trans_term_vec[("may_avoid_dropoff", dropoff)] = 0
            self.default_trans_term_min[("may_avoid_dropoff", dropoff)] = 1


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
            self.actions[origin_node, destination_node] = [Action(trans_min_input, trans_term_vec, trans_term_min, origin_node, destination_node, exog_contrib_vec, cost, {}, {}, [], {})]

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
                partial_trans_min_input[("may_avoid_dropoff", dropoff_node)] = 1
                partial_trans_term_vec[("may_avoid_dropoff", dropoff_node)] = -1

            trans_min_input = ChainMap(partial_trans_min_input, self.default_trans_min_input)
            trans_term_vec = ChainMap(partial_trans_term_vec, self.default_trans_term_vec)
            trans_term_min = ChainMap(partial_trans_term_min, self.default_trans_term_min)

            self.actions[origin_node, destination_node] = [Action(trans_min_input, trans_term_vec, trans_term_min, origin_node, destination_node, exog_contrib_vec, cost, {}, {}, [], {})]
            
        
    def _create_pickup_to_pickup_actions(self):
        for origin_node in self.pickup_to_dropoff:
            for destination_node in self.pickup_to_dropoff:
                if origin_node == destination_node:
                    continue

                cost = self._distance(origin_node, destination_node)
                exog_contrib_vec = self._default_contribution_vector()
                cover_constraint_index = self.rhs_constraint_name_to_index[("cover", origin_node)]
                exog_contrib_vec[cover_constraint_index] = 1
                partial_trans_min_input = {"time": self._travel_time[origin_node, destination_node] + self.service_time[origin_node] + self.service_time[destination_node], 
                                           "volume": self.volume_demands[origin_node] + self.volume_demands[destination_node],
                                           "weight": self.weight_demands[origin_node] + self.weight_demands[destination_node],
                                           "max_combined_loads": 1,
                                           ("may_pickup", destination_node): 1}
                partial_trans_term_vec = {"time": -self._travel_time[origin_node, destination_node] - self.service_time[origin_node], 
                                           "volume": -self.volume_demands[origin_node],
                                           "weight": -self.weight_demands[origin_node],
                                           "max_combined_loads": -1,
                                           ("may_pickup", origin_node): -1,
                                           ("may_avoid_dropoff", self.pickup_to_dropoff[origin_node]): -1}
                partial_trans_term_min = {"time": self.time_window_start[destination_node]}
                trans_min_input = ChainMap(partial_trans_min_input, self.default_trans_min_input)
                trans_term_vec = ChainMap(partial_trans_term_vec, self.default_trans_term_vec)
                trans_term_min = ChainMap(partial_trans_term_min, self.default_trans_term_min)
                
                self.actions[origin_node, destination_node] = [Action(trans_min_input, trans_term_vec, trans_term_min, origin_node, destination_node, exog_contrib_vec, cost, {}, {}, [], {})]
        
    def _create_pickup_to_dropoff_actions(self):
        for origin_node in self.pickup_to_dropoff:
            for destination_node in self.dropoff_to_pickup:
                cost = self._distance(origin_node, destination_node)
                exog_contrib_vec = self._default_contribution_vector()
                cover_constraint_index = self.rhs_constraint_name_to_index[("cover", origin_node)]
                exog_contrib_vec[cover_constraint_index] = 1
                partial_trans_min_input = {"time": self._travel_time[origin_node, destination_node] + self.service_time[origin_node] + self.service_time[destination_node]}
                partial_trans_term_vec = {"time": -self._travel_time[origin_node, destination_node] - self.service_time[origin_node], 
                                           "volume": -self.volume_demands[origin_node],
                                           "weight": -self.weight_demands[origin_node],
                                           "max_combined_loads": -1,
                                           ("may_pickup", origin_node): -1,
                                           ("may_avoid_dropoff", self.pickup_to_dropoff[origin_node]): -1}
                partial_trans_term_min = {"time": self.time_window_start[destination_node]}
                trans_min_input = ChainMap(partial_trans_min_input, self.default_trans_min_input)
                trans_term_vec = ChainMap(partial_trans_term_vec, self.default_trans_term_vec)
                trans_term_min = ChainMap(partial_trans_term_min, self.default_trans_term_min)
                
                self.actions[origin_node, destination_node] = [Action(trans_min_input, trans_term_vec, trans_term_min, origin_node, destination_node, exog_contrib_vec, cost, {}, {}, [], {})]
        
        
    def _create_dropoff_to_pickup_actions(self):
        for origin_node in self.dropoff_to_pickup:
            for destination_node in self.pickup_to_dropoff:
                cost = self._distance(origin_node, destination_node)
                exog_contrib_vec = self._default_contribution_vector()
                partial_trans_min_input = {"time": self._travel_time[origin_node, destination_node] + self.service_time[origin_node] + self.service_time[destination_node], 
                                           "volume": self.volume_demands[destination_node] - self.volume_demands[origin_node],
                                           "weight": self.weight_demands[destination_node] - self.weight_demands[origin_node],
                                           "max_combined_loads": 1,
                                           ("may_pickup", destination_node): 1}
                partial_trans_term_vec = {"time": -self._travel_time[origin_node, destination_node] - self.service_time[origin_node], 
                                           "volume": -self.volume_demands[origin_node],
                                           "weight": -self.weight_demands[origin_node],
                                           ("may_avoid_dropoff", origin_node): 1}
                partial_trans_term_min = {"time": self.time_window_start[destination_node]}
                trans_min_input = ChainMap(partial_trans_min_input, self.default_trans_min_input)
                trans_term_vec = ChainMap(partial_trans_term_vec, self.default_trans_term_vec)
                trans_term_min = ChainMap(partial_trans_term_min, self.default_trans_term_min)
                
                self.actions[origin_node, destination_node] = [Action(trans_min_input, trans_term_vec, trans_term_min, origin_node, destination_node, exog_contrib_vec, cost, {}, {}, [], {})]
        
        
    def _create_dropoff_to_dropoff_actions(self):
        for origin_node in self.dropoff_to_pickup:
            for destination_node in self.dropoff_to_pickup:
                if origin_node == destination_node:
                    continue

                cost = self._distance(origin_node, destination_node)
                exog_contrib_vec = self._default_contribution_vector()
                partial_trans_min_input = {"time": self._travel_time[origin_node, destination_node] + self.service_time[origin_node] + self.service_time[destination_node]}
                partial_trans_term_vec = {"time": -self._travel_time[origin_node, destination_node] - self.service_time[origin_node], 
                                           "volume": -self.volume_demands[origin_node],
                                           "weight": -self.weight_demands[origin_node],
                                           ("may_avoid_dropoff", origin_node): 1}
                partial_trans_term_min = {"time": self.time_window_start[destination_node]}
                trans_min_input = ChainMap(partial_trans_min_input, self.default_trans_min_input)
                trans_term_vec = ChainMap(partial_trans_term_vec, self.default_trans_term_vec)
                trans_term_min = ChainMap(partial_trans_term_min, self.default_trans_term_min)
                
                self.actions[origin_node, destination_node] = [Action(trans_min_input, trans_term_vec, trans_term_min, origin_node, destination_node, exog_contrib_vec, cost, {}, {}, [], {})]
        
        
    def _create_skip_actions(self):
        for destination_node in self.dropoff_to_pickup:
            origin_node = -1
            cost = self._slack(destination_node)
            exog_contrib_vec = self._default_contribution_vector()
            cover_constraint_index = self.rhs_constraint_name_to_index[("cover", origin_node)]
            exog_contrib_vec[cover_constraint_index] = 1

            trans_min_input = ChainMap({}, self.default_trans_min_input)
            trans_term_vec = ChainMap({}, self.default_trans_term_vec)
            trans_term_min = ChainMap({}, self.default_trans_term_min)
            
            self.actions[origin_node, destination_node] = [Action(trans_min_input, trans_term_vec, trans_term_min, origin_node, destination_node, exog_contrib_vec, cost, {}, {}, [], {})]
    
        for origin_node in self.dropoff_to_pickup:
            destination_node = -2
            cost = 0
            exog_contrib_vec = self._default_contribution_vector()

            trans_min_input = ChainMap({}, self.default_trans_min_input)
            trans_term_vec = ChainMap({}, self.default_trans_term_vec)
            trans_term_min = ChainMap({}, self.default_trans_term_min)
            
            self.actions[origin_node, destination_node] = [Action(trans_min_input, trans_term_vec, trans_term_min, origin_node, destination_node, exog_contrib_vec, cost, {}, {}, [], {})]
    

    def _default_contribution_vector(self):
        return zeros(self.number_of_customers)

    def _distance(self, origin, destination):
        x1, y1 = self.coordinates(origin)
        x2, y2 = self.coordinates(destination)
        return hypot(x2 - x1, y2 - y1)

    def _travel_time(self, origin, destination):
        distance = self._distance(origin, destination)
        drive_time = distance / AVERAGE_SPEED
        number_of_rests = int(drive_time / HOS_DRIVE_TIME)
        travel_time = drive_time + number_of_rests * HOS_REST_TIME
        return travel_time
    
    def _slack(self, pickup_node):
        dropoff_node = self.pickup_to_dropoff(pickup_node)
        pickup_dropoff_direct_distance = self._distance(pickup_node, dropoff_node)
        slack_coeff = pickup_dropoff_direct_distance - MIN_DISTANCE_SAVING
        return slack_coeff
        
