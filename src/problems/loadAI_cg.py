from src.problems.optimization_problem import OptimizationProblem
from src.common.action import Action
from src.common.state import State
from src.common.helper import Helper
from src.algorithm.update_states.LoadAI_state_generation_input import LoadAI_state_input
from typing import Dict, List, Tuple, Set
from numpy import zeros, ones, append
from scipy.sparse import csr_matrix
from collections import ChainMap
from math import hypot, radians, sin, cos, sqrt, asin
import pandas as pd
import numpy as np
from numpy import ndarray, array
from collections import defaultdict
import matplotlib.pyplot as plt
from src.algorithm.update_states.state_update_function import StateUpdateFunction
from itertools import permutations
from src.algorithm.cg_solver import GraphMaster_cg
from src.common.route import Route
from src.common.line_time_profiler import HierarchicalProfiler
from src.common.time_profile import TimeProfiler
from tqdm import tqdm
import time
import random
import cProfile
import pstats
import tracemalloc
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
JY_OPT_SPLIT=1
THRESHOLD = [1,10,100, np.inf]
DISTANCE_RATIO = 1
TIME_RATIO = 1
SCORE_RATIO = 1 #best k percent of edge for given shipments
class loadAI_cg:
    def __init__(self, problem_instance_file_name,instance_name, file_type: str = "Standard_Form"):
        
        self.problem_instance_file_name: str = problem_instance_file_name
        self.instance_name = instance_name
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
        self.actions= defaultdict()
        self.initial_res_states: Set[State] = set()
        self.initial_res_actions: Set[Action] = set()
        self.the_single_null_action: Action = None
        self.state_update_module: StateUpdateFunction = None
        self.edges = defaultdict(set)
        self._load_data_from_file()   
        self._build_problem_model()
        self._create_dom_action_object()
        self._create_initial_res_states()
        self._create_initial_res_actions()
        self._define_state_update_module()
        self._get_benefit_group()
        # path = 'Load_ai.xlsx'
        # self._evaluate_solution(path)
        #self.plot_pickup_dropoff_with_clusters()
    def solve(self):
        """Creates a GraphMasterSolver instance from problem data and calls its solve() method"""
        #node_to_list = self._group_states_by_node_l(self.initial_res_states)
        self.solver = GraphMaster_cg(
            self.nodes,
            self.actions,
            self.can_group,
            self.edges,
            self.preferred_actions,
            self.distance,
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
            self.neighbors,
            self.benefit_group,
            self.benefit_group_cost
            #node_to_list
        )
        #with TimeProfiler(f'time_profile_{self.instance_name}_speed'):
        profiler = cProfile.Profile()
        profiler.enable()

        output = self.solver.solve()

        profiler.disable()
        profiler.dump_stats('program_profile_400_x.prof')

        variable_to_values = output['x']
        routes:List[Route] = output['used_routes']
        output_info = output['output_info']
        opt_gap = output['optimality_gap']
        print('=======output info=======')
        for name,value in output_info.items():
            print(' ')
            print(name)
            print(value)
        print(f'optimality gap: {opt_gap*100}%')
        import csv
        filename="output.csv"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['sequence_of_nodes', 'cost'])
            for route in routes:
                sequece_of_node = route.node_in_ordered
                cost = route.cost
                sequence_str = ','.join(map(str, sequece_of_node))
            
                # Write row with sequence (as string) and cost
                writer.writerow([sequence_str, cost])
        print(f"Routes successfully saved to {filename}")
            


    
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
        self.pickup_node = []
        self.dropoff_node = []
        self.problem_info = {}
        self.problem_info['volume_capacity'] = VOLUME_CAPACITY
        self.problem_info['weight_capacity'] = WEIGHT_CAPACITY
        self.problem_info['hos_drive_time'] = HOS_DRIVE_TIME
        self.problem_info['hos_work_time'] = HOS_WORK_TIME
        self.problem_info['max_combined_loads'] = MAX_COMBINED_LOADS
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
            self.pickup_node.append(pickup_id)
            self.dropoff_to_pickup[dropoff_id] = pickup_id
            self.dropoff_node.append(dropoff_id)
        self.pickup_and_dropoff_node = self.pickup_node+self.dropoff_node
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
        #self._create_travel_time()
        #print('check here')
        self.rhs_vector = ones(self.number_of_customers)
        idx = 0
        for pickup_node in self.pickup_node:
            self.rhs_dict[str(("Cover", pickup_node))] = 1
            self.rhs_constraint_name_to_index[str(("Cover", pickup_node))] = idx
            self.rhs_index_to_constraint_name[idx] = str(("Cover", pickup_node))
            idx += 1

        # INITIAL RESOURCE STATE
        self._populate_initial_resources()

        # ACTIONS
        self._create_default_resource_values()
        self.travel_time_hos, self.distance, self.travel_time = self._travel_time_and_distance()
        create_edge_pair_first = True
        if create_edge_pair_first is False:
            with TimeProfiler(f'time_profile_edges_creation'):
                self._create_edges()
                breakpoint()
        else:
            self._create_edges_pair()
            # print(f'{len(self.pairs)} edges generated')
            #breakpoint()
            
            #profiler = cProfile.Profile()

            # Start profiling
            #profiler.enable()
            #tracemalloc.start()
            #with TimeProfiler(f'time_profile_edges_creation'):
            self._create_actions_with_edge_pair()
            #snapshot = tracemalloc.take_snapshot()
            #top_stats = snapshot.statistics('lineno')

            # Print the top 10 memory-consuming lines
            #print("[ Top 10 memory usage ]")
            # for stat in top_stats[:10]:
            #     print(stat)
            #profiler.disable()

            # Save results to a file
            #profiler.dump_stats('program_profile.prof')

            # Print the top 10 time-consuming functions
            #stats = pstats.Stats('program_profile.prof')
        
        #breakpoint()
        #self._create_edges_pair()
        self._create_source_sink_actions()
        # self._create_pickup_to_pickup_actions()
        # self._create_pickup_to_dropoff_actions()
        # self._create_dropoff_to_pickup_actions()
        # self._create_dropoff_to_dropoff_actions()
        self._create_skip_actions()
        self._create_null_action_info()
        self._create_preferred_actions()
        #breakpoint()
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

        self.initial_resource_dict["how_drive"] = self.max_combined_loads
        self.initial_resource_vector = append(self.initial_resource_vector, self.hos_drive_time)
        self.resource_name_to_index["hos_drive"] = idx
        self.resource_index_to_name[idx] = "hos_drive"
        idx += 1

        self.initial_resource_dict["how_work"] = self.max_combined_loads
        self.initial_resource_vector = append(self.initial_resource_vector, self.hos_work_time)
        self.resource_name_to_index["how_work"] = idx
        self.resource_index_to_name[idx] = "how_work"
        idx += 1


        self.number_of_resources = len(self.initial_resource_dict)
        #self.initial_resource_vector=csr_matrix(self.initial_resource_vector.reshape(1, -1))

    def _empty_resource_vec(self) -> csr_matrix:
        return self.empty_resource_array
    
    def _full_resource_vec(self):
        return self.full_resource_array


    def _create_default_resource_values(self):
        """Defines default resource values for actions."""
        self.empty_resource_array = zeros(self.number_of_resources)

        self.default_trans_min_input = {
            "weight": 0,
            "volume": 0,
            "time": 0,
            "max_combined_loads": 0,
            'hos_drive':0,
            'hos_work':0
        }
        self.default_trans_term_vec = {
            "weight": 0,
            "volume": 0,
            "time": 0,
            "max_combined_loads": 0,
            'hos_drive':0,
            'hos_work':0
        }
        self.default_trans_term_min = {
            "weight": self.weight_capacity,
            "volume": self.volume_capacity,
            "time": self.maximum_time,
            "max_combined_loads": self.max_combined_loads,
            'hos_drive':self.hos_drive_time,
            'how_work':self.hos_work_time
        }

        # for pickup in self.pickup_to_dropoff:
        #     self.default_trans_min_input[str(("may_pickup", pickup))] = 0
        #     self.default_trans_term_vec[str(("may_pickup", pickup))] = 0
        #     self.default_trans_term_min[str(("may_pickup", pickup))] = 1

        # for dropoff in self.dropoff_to_pickup:
        #     self.default_trans_min_input[str(("may_avoid_dropoff", dropoff))] = 0
        #     self.default_trans_term_vec[str(("may_avoid_dropoff", dropoff))] = 0
        #     self.default_trans_term_min[str(("may_avoid_dropoff", dropoff))] = 1

        self.full_resource_array = zeros(len(self.resource_name_to_index))

        for resource_name, index in self.resource_name_to_index.items():
            if resource_name in self.default_trans_term_min:
                self.full_resource_array[index] = self.default_trans_term_min[resource_name]


    def _create_source_sink_actions(self):
        for destination_node in tqdm(self.pickup_node,desc = 'create_source_sink_actions_1'):
            origin_node = -1  # Source
            self.edges[origin_node].add(destination_node)
            cost = 0
            #exog_contrib_vec = self._default_contribution_vector()
            non_zero_exog_val = None
            non_zero_exog_indices = None

            min_resource_vec = np.array([0,0,0,0,0,0])
            resource_consumption_vec = np.array([0,0,0,0,0,0])
            max_resource_vec = np.array([WEIGHT_CAPACITY,VOLUME_CAPACITY,self.time_window_start[destination_node],MAX_COMBINED_LOADS,self.hos_drive_time,self.hos_work_time])

            this_pickup = None
            this_dropoff = None
            #indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,trans_term_min)
            #indices_apply_min_to=Helper.LOAD_AI_partial_map_2_indices_applied(self.resource_name_to_index,self.pickup_node,self.dropoff_node,destination_node,origin_node,self.number_of_customers)

            action = Action(destination_node, origin_node, this_pickup, this_dropoff, self.number_of_customers, non_zero_exog_val,non_zero_exog_indices,cost,min_resource_vec, 
                            resource_consumption_vec,max_resource_vec,[self.time_window_start[destination_node],self.time_window_end[destination_node]],
                            0,0)
            self.actions[origin_node, destination_node] = [action]

        for origin_node in tqdm(self.dropoff_node,desc='create_source_sink_actions_2'):
            origin_node = origin_node
            destination_node = -2  # Sink
            self.edges[origin_node].add(destination_node)
            cost = 0
            #exog_contrib_vec = self._default_contribution_vector()
            if JY_OPT_SPLIT==1:
                cover_constraint_index = self.rhs_constraint_name_to_index[str(("Cover", self.dropoff_to_pickup[origin_node]))]
                #exog_contrib_vec[cover_constraint_index] = 0.5
                non_zero_exog_val = 0.5
                non_zero_exog_indices = cover_constraint_index
                #print('cover_constraint_index')
                #print(cover_constraint_index)
                #print('origin_node')
                #print(origin_node)
                #print('self.dropoff_to_pickup[origin_node]')
                #print(self.dropoff_to_pickup[origin_node])
                #input('----')

            min_resource_vec = np.array([0,0,0,0,0,0])
            resource_consumption_vec = np.array([0,0,0,0,0,0])
            max_resource_vec = np.array([WEIGHT_CAPACITY,VOLUME_CAPACITY,self.maximum_time,MAX_COMBINED_LOADS,self.hos_drive_time,self.hos_work_time])
            
            #indices_apply_min_to=Helper.LOAD_AI_partial_map_2_indices_applied(self.resource_name_to_index,self.pickup_node,self.dropoff_node,destination_node,origin_node,self.number_of_customers)
            this_pickup = None
            this_dropoff = origin_node - self.number_of_customers
            action = Action(destination_node, origin_node, this_pickup,this_dropoff,self.number_of_customers,
                             non_zero_exog_val,non_zero_exog_indices,cost, min_resource_vec, 
                            resource_consumption_vec,max_resource_vec,[self.maximum_time,0],0,0)
        
            self.actions[origin_node, destination_node] = [action]

    def _create_pickup_to_pickup_actions(self, origin_node,destination_node):
        if origin_node == destination_node:
            return None
        travel_time = self.travel_time[origin_node,destination_node]
        distance = self.distance[origin_node, destination_node]
        self.edges[origin_node].add(destination_node)
        cost = distance

        cover_constraint_index = self.rhs_constraint_name_to_index[str(("Cover", origin_node))]
        non_zero_exog_val = 1.0

        
        non_zero_exog_indices = cover_constraint_index
        if JY_OPT_SPLIT==1:
            non_zero_exog_val = 0.5

        min_resource_vec = np.array([self.weight_demands[origin_node] + self.weight_demands[destination_node],
                                     self.volume_demands[origin_node] + self.volume_demands[destination_node],
                                     travel_time + self.service_time[origin_node] + self.time_window_end[destination_node],
                                     1,0,0])
        resource_consumption_vec = np.array([-self.weight_demands[origin_node],
                                     -self.volume_demands[origin_node],
                                     -travel_time,
                                     -1,0,0])
        max_resource_vec = np.array([WEIGHT_CAPACITY,VOLUME_CAPACITY,
                                             self.time_window_start[destination_node],
                                             MAX_COMBINED_LOADS,self.hos_drive_time,self.hos_work_time] )     
        #indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,partial_trans_term_min)
        #indices_apply_min_to=Helper.LOAD_AI_partial_map_2_indices_applied(self.resource_name_to_index,self.pickup_node,self.dropoff_node,destination_node,origin_node,self.number_of_customers)
        this_pick_up = origin_node
        this_drop_off = None
        action = Action(destination_node, origin_node, this_pick_up,this_drop_off, self.number_of_customers,
                             non_zero_exog_val,non_zero_exog_indices,cost, min_resource_vec,resource_consumption_vec, 
                            max_resource_vec, [self.time_window_start[destination_node],self.time_window_end[destination_node]],
                            travel_time, self.service_time[origin_node])
        self.actions[(origin_node, destination_node)] = [action]
        #return action
        
    def _create_pickup_to_dropoff_actions(self,origin_node, destination_node):
        travel_time = self.travel_time[origin_node,destination_node]
        distance = self.distance[origin_node,destination_node]
        self.edges[origin_node].add(destination_node)
        cost = distance
        #exog_contrib_vec = self._default_contribution_vector()
        cover_constraint_index = self.rhs_constraint_name_to_index[str(("Cover", origin_node))]
        non_zero_exog_val = None
        #exog_contrib_vec[cover_constraint_index] = 1
        
        non_zero_exog_indices = cover_constraint_index
        if JY_OPT_SPLIT==1:
            non_zero_exog_val = 0.5
            #exog_contrib_vec[cover_constraint_index] = 0.5

        min_resource_vec = np.array([0,0,travel_time + self.service_time[origin_node] + self.time_window_end[destination_node],0,0,0])
        resource_consumption_vec = np.array([-self.weight_demands[origin_node],-self.volume_demands[origin_node],
                                    -travel_time ,
                                    0,0,0])
        max_resource_vec = np.array([WEIGHT_CAPACITY,VOLUME_CAPACITY,self.time_window_start[destination_node],MAX_COMBINED_LOADS,self.hos_drive_time,self.hos_work_time])
        
        # min_resource_vec_indices,min_resource_vec_data = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_min_input)     
        # resource_consumption_vec_indices,resource_consumption_vec_data = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_term_vec)     
        # max_resource_vec_indices,max_resource_vec_data = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_trans_term_min)     
        #indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,partial_trans_term_min)
        #indices_apply_min_to=Helper.LOAD_AI_partial_map_2_indices_applied(self.resource_name_to_index,self.pickup_node,self.dropoff_node,destination_node,origin_node,self.number_of_customers)
        this_pick_up = origin_node
        this_drop_off = None
        action = Action( destination_node, origin_node, this_pick_up,this_drop_off,self.number_of_customers,
                             non_zero_exog_val,non_zero_exog_indices,cost, min_resource_vec, 
                            resource_consumption_vec, max_resource_vec, [self.time_window_start[destination_node],self.time_window_end[destination_node]],
                            travel_time, self.service_time[origin_node])
        self.actions[(origin_node, destination_node)] = [action]
        #return action
        
    def _create_dropoff_to_pickup_actions(self,origin_node, destination_node):

        origin_pickup_node = self.dropoff_to_pickup[origin_node]
        if origin_pickup_node == destination_node:
            return None

        travel_time = self.travel_time[origin_node,destination_node]
        distance = self.distance[origin_node, destination_node]
        self.edges[origin_node].add(destination_node)
        cost = distance
        #exog_contrib_vec = self._default_contribution_vector()
        if JY_OPT_SPLIT==1:
            cover_constraint_index = self.rhs_constraint_name_to_index[str(("Cover", self.dropoff_to_pickup[origin_node]))]
            non_zero_exog_val = 0.5
            #exog_contrib_vec[cover_constraint_index] = 0.5
            non_zero_exog_indices = cover_constraint_index
        min_resource_vec = np.array([self.weight_demands[destination_node] - self.weight_demands[origin_pickup_node],
                                    self.volume_demands[destination_node] - self.volume_demands[origin_pickup_node],
                                    travel_time + self.service_time[origin_node] + self.time_window_end[destination_node], 
                                    1,0,0])
        resource_consumption_vec =np.array([-self.weight_demands[origin_pickup_node],
                                    -self.volume_demands[origin_pickup_node],
                                    -travel_time , 
                                    -1,0,0])
        max_resource_vec = np.array([WEIGHT_CAPACITY,VOLUME_CAPACITY,
                                    self.time_window_start[destination_node],MAX_COMBINED_LOADS,self.hos_drive_time,self.hos_work_time])
        # trans_min_input = ChainMap(partial_trans_min_input, self.default_trans_min_input)
        # trans_term_vec = ChainMap(partial_trans_term_vec, self.default_trans_term_vec)
        # trans_term_min = ChainMap(partial_trans_term_min, self.default_trans_term_min)
        
 
        #indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,partial_trans_term_min)
        #indices_apply_min_to=Helper.LOAD_AI_partial_map_2_indices_applied(self.resource_name_to_index,self.pickup_node,self.dropoff_node,destination_node,origin_node,self.number_of_customers)
        this_pickup = None
        this_dropoff= origin_node - self.number_of_customers
        action = Action(destination_node, origin_node, this_pickup,this_dropoff,self.number_of_customers,
                            non_zero_exog_val, non_zero_exog_indices, cost, min_resource_vec, 
                            resource_consumption_vec, 
                            max_resource_vec,[self.time_window_start[destination_node],self.time_window_end[destination_node]],
                            travel_time,self.service_time[origin_node])
        self.actions[(origin_node, destination_node)] = [action]
        #return action
        
    def _create_dropoff_to_dropoff_actions(self, origin_node, destination_node):
        if origin_node == destination_node:
            return None
        travel_time = self.travel_time[origin_node,destination_node]
        distance = self.distance[origin_node, destination_node]
        self.edges[origin_node].add(destination_node)
        cost = distance
        origin_pickup_node = self.dropoff_to_pickup[origin_node]
        #exog_contrib_vec = self._default_contribution_vector()
        if JY_OPT_SPLIT==1:
            cover_constraint_index = self.rhs_constraint_name_to_index[str(("Cover", self.dropoff_to_pickup[origin_node]))]
            non_zero_exog_val = 0.5
            #exog_contrib_vec[cover_constraint_index] = 0.5
            non_zero_exog_indices = cover_constraint_index

        min_resource_vec = np.array([0,0,travel_time + self.service_time[origin_node] + self.time_window_end[destination_node],0,0,0])
        resource_consumption_vec = np.array([ -self.weight_demands[origin_pickup_node],-self.volume_demands[origin_pickup_node],
                                 -travel_time , 0,0,0])
        max_resource_vec = np.array([WEIGHT_CAPACITY,VOLUME_CAPACITY, self.time_window_start[destination_node],MAX_COMBINED_LOADS,self.hos_drive_time,self.hos_work_time])

        
        
        #indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,partial_trans_term_min)
        #indices_apply_min_to=Helper.LOAD_AI_partial_map_2_indices_applied(self.resource_name_to_index,self.pickup_node,self.dropoff_node,destination_node,origin_node,self.number_of_customers)
        this_pickup = None
        this_dropoff = origin_node - self.number_of_customers
        action = Action(destination_node, origin_node, this_pickup,this_dropoff,self.number_of_customers,
                            non_zero_exog_val , non_zero_exog_indices, cost, min_resource_vec, 
                            resource_consumption_vec, 
                            max_resource_vec,[self.time_window_start[destination_node],self.time_window_end[destination_node]],
                            travel_time, self.service_time[origin_node])
        self.actions[(origin_node, destination_node)] = [action]
        #return action
        
    def _create_skip_actions(self):
        for destination_node in tqdm(self.pickup_node,desc='create_skip_actions'):
            origin_node = -1
            self.edges[origin_node].add(destination_node + 2 * self.number_of_customers)

            cost = self._slack(destination_node)
            #exog_contrib_vec = self._default_contribution_vector()
            cover_constraint_index = self.rhs_constraint_name_to_index[str(("Cover", destination_node))]
            #exog_contrib_vec[cover_constraint_index] = 1
            non_zero_exog_val = 1
            non_zero_exog_indices = cover_constraint_index
            # trans_min_input = ChainMap({}, self.default_trans_min_input)
            # trans_term_vec = ChainMap({}, self.default_trans_term_vec)
            # trans_term_min = ChainMap({}, self.default_trans_term_min)
            min_resource_vec = np.array([0,0,0,0,0,0])
            resource_consumption_vec = np.array([0,0,0,0,0,0])
            max_resource_vec= np.array([WEIGHT_CAPACITY,VOLUME_CAPACITY,self.maximum_time, MAX_COMBINED_LOADS,self.hos_drive_time,self.hos_work_time])
            
            #indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,{})
            #indices_apply_min_to=Helper.LOAD_AI_partial_map_2_indices_applied(self.resource_name_to_index,self.pickup_node,self.dropoff_node,destination_node,origin_node,self.number_of_customers)
            this_pickup = destination_node
            this_dropoff = None
            time_window = [self.time_window_start[destination_node],self.time_window_end[destination_node]]
            destination_node = destination_node + 2 * self.number_of_customers
            action = Action( destination_node, origin_node, this_pickup,this_dropoff,self.number_of_customers,
                            non_zero_exog_val ,non_zero_exog_indices,cost, min_resource_vec, 
                            resource_consumption_vec, 
                            max_resource_vec,time_window,0,0)
        
            self.actions[origin_node, destination_node] = [action]
               
        for origin_node in self.pickup_node:
            destination_node = -2
            self.edges[origin_node + 2 * self.number_of_customers].add(destination_node)
            cost = 0
            #exog_contrib_vec = self._default_contribution_vector()
            non_zero_exog_val = None
            non_zero_exog_indices = None
            min_resource_vec = np.array([0,0,0,0,0,0])
            resource_consumption_vec = np.array([0,0,0,0,0,0])
            max_resource_vec= np.array([WEIGHT_CAPACITY,VOLUME_CAPACITY,self.maximum_time, MAX_COMBINED_LOADS,self.hos_drive_time,self.hos_work_time])
            
            
            #indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,{})
            this_pickup = None
            this_dropoff = origin_node
            origin_node = origin_node + 2 * self.number_of_customers
            #indices_apply_min_to=Helper.LOAD_AI_partial_map_2_indices_applied(self.resource_name_to_index,self.pickup_node,self.dropoff_node,destination_node,origin_node,self.number_of_customers)

            action = Action(destination_node, origin_node, this_pickup,this_dropoff,self.number_of_customers,
                             non_zero_exog_val,non_zero_exog_indices,cost, min_resource_vec, 
                            resource_consumption_vec, 
                            max_resource_vec,[self.maximum_time,0],0,0)
            self.actions[origin_node, destination_node] = [action]
        
    def _ez_check_for_create_group(self,overlap,threshold,u,v):
        time = self.travel_time[u,v]
        dist = self.distance[u,v]
        if dist < DISTANCE_RATIO*threshold and time < TIME_RATIO*overlap:
            return True
        else:
            return False
    def _create_group(self):
        group = defaultdict(list)
        for u in tqdm(self.pickup_node,desc='create initial group'):
            for v in self.pickup_node:
                if u!=v:
                    time_window_end_dropoff_u = self.time_window_end[self.pickup_and_dropoff_node[u]]
                    time_window_start_pickup_v = self.time_window_start[v]
                    time_window_end_dropoff_v = self.time_window_end[self.pickup_and_dropoff_node[v]]
                    time_window_start_pickup_u = self.time_window_start[u]
                    overlap_1 = time_window_start_pickup_v-time_window_end_dropoff_u
                    overlap_2 = time_window_start_pickup_u-time_window_end_dropoff_v
                    overlap = min(overlap_1,overlap_2)
                    threshold = max(self.distance[u,self.pickup_to_dropoff[u]],self.distance[v,self.pickup_to_dropoff[v]])
                    minDist = min(self.distance[u,v],self.distance[u,self.pickup_to_dropoff[v]],self.distance[self.pickup_to_dropoff[u],v],self.distance[self.pickup_to_dropoff[u],self.pickup_to_dropoff[v]])
                    minTime = max(self.travel_time[u,v],self.travel_time[u,self.pickup_to_dropoff[v]],self.travel_time[self.pickup_to_dropoff[u],v],self.travel_time[self.pickup_to_dropoff[u],self.pickup_to_dropoff[v]]\
                                  ,self.travel_time[u,self.pickup_to_dropoff[u]],self.travel_time[v,self.pickup_to_dropoff[v]])
                    if overlap>0:
                        if minDist<DISTANCE_RATIO*threshold and minTime<TIME_RATIO*min(overlap_1,overlap_2):
                            group[u].append(v)
                    else:
                        time_u_v = self.travel_time[self.pickup_to_dropoff[u],v]
                        time_v_u = self.travel_time[self.pickup_to_dropoff[v],u]
                        if overlap_1 <0 and time_u_v + overlap_1<0:
                            group[u].append(v)
                        if overlap_2 <0 and time_v_u+overlap_2<0:
                            group[u].append(v)

        return group
    def _create_edges(self):
        initial_group = self._create_group()
        self.can_group = defaultdict(set)
        
        for u in self.pickup_node:
            self._create_pickup_to_dropoff_actions(u,self.pickup_to_dropoff[u])

        for u in tqdm(self.pickup_node,desc='creating edges'):
            for v in initial_group[u]:

                did_create_edge = False
                earlist_time_arrive_v_pickup = self.time_window_start[u]-self.service_time[u]-self.travel_time[u,v]

                if earlist_time_arrive_v_pickup>self.time_window_end[v]:
                    earlist_time_depart_v_pickup = min(self.time_window_start[v],earlist_time_arrive_v_pickup)
                    #t1_2,d1_2 = self._travel_time_and_distance(v,self.pickup_to_dropoff[u])
                    earlist_time_arrive_u_dropoff = earlist_time_depart_v_pickup-self.service_time[v]\
                        -self.travel_time[v,self.pickup_to_dropoff[u]]
                    #t2_2,d2_2 = self._travel_time_and_distance(v,self.pickup_to_dropoff[v])
                    earlist_time_arrive_v_dropoff = earlist_time_depart_v_pickup - self.service_time[v]\
                        -self.travel_time[v,self.pickup_to_dropoff[v]]
                    if earlist_time_arrive_u_dropoff > self.time_window_end[self.pickup_to_dropoff[u]]:
                        earlist_time_depart_u_dropoff = min(self.time_window_start[self.pickup_to_dropoff[u]],earlist_time_arrive_u_dropoff)
                        #t1_3,d1_3 = self._travel_time_and_distance(self.pickup_to_dropoff[u],self.pickup_to_dropoff[v])
                        earlist_time_arrive_v_dropoff_2 = earlist_time_depart_u_dropoff - self.service_time[u]-self.travel_time[self.pickup_to_dropoff[u],self.pickup_to_dropoff[v]]
                        if earlist_time_arrive_v_dropoff_2 > self.time_window_end[self.pickup_to_dropoff[v]]:
                            did_create_edge = True
                            if (u,v) not in self.actions:
                                this_action = self._create_pickup_to_pickup_actions(u,v)
                            if (v,self.pickup_to_dropoff[u]) not in self.actions:
                                this_action = self._create_pickup_to_dropoff_actions(v,self.pickup_to_dropoff[u])
                            if (self.pickup_to_dropoff[u],self.pickup_to_dropoff[v]) not in self.actions:
                                this_action = self._create_dropoff_to_dropoff_actions(self.pickup_to_dropoff[u],self.pickup_to_dropoff[v])

                            

                    if earlist_time_arrive_v_dropoff>self.time_window_end[self.pickup_to_dropoff[v]]:
                        earlist_time_depart_v_dropoff = min(self.time_window_start[self.pickup_to_dropoff[v]],earlist_time_arrive_v_dropoff)
                        #t2_3,d2_3 = self._travel_time_and_distance(self.pickup_to_dropoff[v],self.pickup_to_dropoff[u])
                        earlist_time_arrive_u_dropoff_2 = earlist_time_depart_v_dropoff -self.service_time[v]-self.travel_time[self.pickup_to_dropoff[v],self.pickup_to_dropoff[u]]
                        if earlist_time_arrive_u_dropoff_2 > self.time_window_end[self.pickup_to_dropoff[u]]:
                            did_create_edge= True
                            if (u,v) not in self.actions:
                                this_action = self._create_pickup_to_pickup_actions(u,v)
                            if (v,self.pickup_to_dropoff[v]) not in self.actions:
                                this_action = self._create_pickup_to_dropoff_actions(v,self.pickup_to_dropoff[v])
                            if (self.pickup_to_dropoff[v],self.pickup_to_dropoff[u]) not in self.actions:
                                this_action = self._create_dropoff_to_dropoff_actions(self.pickup_to_dropoff[v],self.pickup_to_dropoff[u])

                earlist_time_arrive_dropoff_u_3 = self.time_window_start[u] - self.service_time[u]-self.travel_time[u, self.pickup_to_dropoff[u]]
                earlist_time_depart_dropoff_u_3 = min(self.time_window_start[self.pickup_to_dropoff[u]],earlist_time_arrive_dropoff_u_3)
                #t3,d4 = self._travel_time_and_distance(self.pickup_to_dropoff[u],v)
                earlist_time_arrive_pickup_v_3 = earlist_time_depart_dropoff_u_3 - self.service_time[u]-self.travel_time[self.pickup_to_dropoff[u],v]
                if earlist_time_arrive_pickup_v_3 > self.time_window_start[v]:
                    did_create_edge = True
                    if (self.pickup_to_dropoff[u],v) not in self.actions:
                        this_action = self._create_dropoff_to_pickup_actions(self.pickup_to_dropoff[u],v)

                if did_create_edge == True:
                    self.can_group[u].add(v)
    def _create_edges_pair(self):
        #initial_group = self._create_group()
        self.can_group = defaultdict(set)
        self.pickup_pickup_pairs = set()
        self.pickup_dropoff_pairs = set()
        self.dropoff_pickup_pairs = set()
        self.dropoff_dropoff_pairs = set()
        self.action_pair_to_actions = defaultdict(lambda: defaultdict(tuple))
        for u in self.pickup_node:
            #self._create_pickup_to_dropoff_actions(u,self.pickup_to_dropoff[u])
            self.pickup_dropoff_pairs.add((u,self.pickup_to_dropoff[u]))

        for u in tqdm(self.pickup_node,desc='creating edges pairs'):
            for v in self.pickup_node:
                if u != v:
                    did_create_edge = False
                    earlist_time_arrive_v_pickup = self.time_window_start[u]-self.service_time[u]-self.travel_time_hos[u,v]
                    time_freedom_uv = earlist_time_arrive_v_pickup - self.time_window_end[v]
                    if time_freedom_uv>0:
                        earlist_time_depart_v_pickup = min(self.time_window_start[v],earlist_time_arrive_v_pickup)
                        #t1_2,d1_2 = self._travel_time_and_distance(v,self.pickup_to_dropoff[u])
                        earlist_time_arrive_u_dropoff = earlist_time_depart_v_pickup-self.service_time[v]\
                            -self.travel_time_hos[v,self.pickup_to_dropoff[u]]
                        time_freedom_uvu = earlist_time_arrive_u_dropoff - self.time_window_end[self.pickup_to_dropoff[u]]
                        #t2_2,d2_2 = self._travel_time_and_distance(v,self.pickup_to_dropoff[v])
                        earlist_time_arrive_v_dropoff = earlist_time_depart_v_pickup - self.service_time[v]\
                            -self.travel_time_hos[v,self.pickup_to_dropoff[v]]
                        time_freedom_uvv = earlist_time_arrive_v_dropoff-self.time_window_end[self.pickup_to_dropoff[v]]
                        if time_freedom_uvu>0:
                            earlist_time_depart_u_dropoff = min(self.time_window_start[self.pickup_to_dropoff[u]],earlist_time_arrive_u_dropoff)
                            #t1_3,d1_3 = self._travel_time_and_distance(self.pickup_to_dropoff[u],self.pickup_to_dropoff[v])
                            earlist_time_arrive_v_dropoff_2 = earlist_time_depart_u_dropoff - self.service_time[u]-self.travel_time_hos[self.pickup_to_dropoff[u],self.pickup_to_dropoff[v]]
                            time_freedom_uvuv = earlist_time_arrive_v_dropoff_2-self.time_window_end[self.pickup_to_dropoff[v]]
                            if time_freedom_uvuv>0:
                                did_create_edge = True

                                this_time_freedom = min(time_freedom_uv,time_freedom_uvu,time_freedom_uvuv)
                                self.action_pair_to_actions[u][this_time_freedom]=[1,u,v,self.pickup_to_dropoff[u],self.pickup_to_dropoff[v]]

                        if time_freedom_uvv>0:
                            earlist_time_depart_v_dropoff = min(self.time_window_start[self.pickup_to_dropoff[v]],earlist_time_arrive_v_dropoff)
                            #t2_3,d2_3 = self._travel_time_and_distance(self.pickup_to_dropoff[v],self.pickup_to_dropoff[u]
                            earlist_time_arrive_u_dropoff_2 = earlist_time_depart_v_dropoff -self.service_time[v]-self.travel_time_hos[self.pickup_to_dropoff[v],self.pickup_to_dropoff[u]]
                            time_freedom_uvvu = earlist_time_arrive_u_dropoff_2 - self.time_window_end[self.pickup_to_dropoff[u]]
                            if time_freedom_uvvu>0:
                                did_create_edge= True

                                this_time_freedom = min(time_freedom_uv,time_freedom_uvv,time_freedom_uvvu)
                                self.action_pair_to_actions[u][this_time_freedom]=[1,u,v,self.pickup_to_dropoff[v],self.pickup_to_dropoff[u]]
                    earlist_time_arrive_dropoff_u_3 = self.time_window_start[u] - self.service_time[u]-self.travel_time_hos[u, self.pickup_to_dropoff[u]]
                    time_freedom_uu = earlist_time_arrive_dropoff_u_3 - self.time_window_end[self.pickup_to_dropoff[u]]
                    earlist_time_depart_dropoff_u_3 = min(self.time_window_start[self.pickup_to_dropoff[u]],earlist_time_arrive_dropoff_u_3)
                    
                    earlist_time_arrive_pickup_v_3 = earlist_time_depart_dropoff_u_3 - self.service_time[u]-self.travel_time_hos[self.pickup_to_dropoff[u],v]
                    time_freedom_uuv = earlist_time_arrive_pickup_v_3 - self.time_window_end[v]
                    if time_freedom_uuv >0:
                        did_create_edge = True

                        earlist_time_depart_pickup_v_3 = min(self.time_window_start[v],earlist_time_arrive_pickup_v_3)
                        earlist_time_arrive_dropoff_v_3 = earlist_time_depart_pickup_v_3 - self.service_time[v]-self.travel_time_hos[v,self.pickup_to_dropoff[v]]
                        time_freedom_uuvv = earlist_time_arrive_dropoff_v_3 - self.time_window_end[self.pickup_to_dropoff[v]]
                        this_time_freedom = min(time_freedom_uu,time_freedom_uuv,time_freedom_uuvv)
                        self.action_pair_to_actions[u][this_time_freedom] = [2,u,self.pickup_to_dropoff[u],v,self.pickup_to_dropoff[v]]
                    if did_create_edge == True:
                        self.can_group[u].add(v)
        self._get_best_edge_by_score()
        print(f'total {len(self.pickup_pickup_pairs)+len(self.pickup_dropoff_pairs)+len(self.dropoff_pickup_pairs)+len(self.dropoff_dropoff_pairs)} generated')
        print('check here')
        #breakpoint()
    def _get_best_edge_by_score(self):
        for u,this_dict in self.action_pair_to_actions.items():
            sorted_dict = dict(sorted(this_dict.items(), key=lambda item: item[1], reverse=True))
            best_k = int(len(sorted_dict) * SCORE_RATIO)
            for key, value in list(sorted_dict.items())[:best_k]:
                if value[0] == 1:
                    self.pickup_pickup_pairs.add((value[1],value[2]))
                    self.pickup_dropoff_pairs.add((value[2],value[3]))
                    self.dropoff_dropoff_pairs.add((value[3],value[4]))
                if value[0] == 2:
                    self.dropoff_pickup_pairs.add((value[2],value[3]))
    def _create_actions_with_edge_pair(self):
        print('create edge from pairs')
                
        for (u,v) in tqdm(self.pickup_pickup_pairs, desc='create action for pickup to pickup'):
            self._create_pickup_to_pickup_actions(u,v)
        for (u,v) in tqdm(self.pickup_dropoff_pairs, desc='create action for pickup to dropoff'):
            self._create_pickup_to_dropoff_actions(u,v)
        for (u,v) in tqdm(self.dropoff_pickup_pairs, desc='create action for dropoff to pickup'):
            self._create_dropoff_to_pickup_actions(u,v)
        for (u,v) in tqdm(self.dropoff_dropoff_pairs, desc='create action for dropoff to dropoff'):
            self._create_dropoff_to_dropoff_actions(u,v)
        # for (u,v) in tqdm(self.pairs,desc='creating action based on edge pair'):
        #     if u in self.pickup_node and v in self.pickup_node:
        #         self._create_pickup_to_pickup_actions(u,v)
        #     if u in self.pickup_node and v in self.dropoff_node:
        #         self._create_pickup_to_dropoff_actions(u,v)
        #     if u in self.dropoff_node and v in self.pickup_node:
        #         self._create_dropoff_to_pickup_actions(u,v)
        #     if u in self.dropoff_node and v in self.dropoff_node:
        #         self._create_dropoff_to_dropoff_actions(u,v)
    def _create_preferred_actions(self):
        #self.parefered_actions = {threshold:[] for threshold in THRESHOLD}
        self.preferred_actions = {}
        self.dict = self.distance.copy()
        print('====start create preferred edges====')
        use_feasible_edge = True
        if use_feasible_edge is True:
            self.preferred_actions[np.inf] = self.edges.copy()
        else:
            for threshold in THRESHOLD:

                F = {2:{}}
                B = {2:{}}
                for u in self.nodes:
                    F[2][u] = set()
                    for v in self.edges[u]:
                        if v not in B[2]:    # ensure v is initialized
                            B[2][v] = set()
                        if self.dict[(u,v)] <= threshold:
                            F[2][u].add(v)
                            B[2][v].add(u)

                for k in range(3,MAX_COMBINED_LOADS+1):
                    F[k]={}
                    B[k]={}

                    for u in self.nodes:
                        for v in self.edges[u]:
                            min_dist = self.dict[(u,v)]
                            for w in set(F[k-1].get(u,set())) & set(B[k-1].get(v,set())):
                                if self.dict[(u,w)] + self.dict[(w,v)] < min_dist:
                                    min_dist = self.dict[(u,w)] + self.dict[(w,v)]
                            self.dict[(u,v)] = min_dist

                    for u in self.nodes:
                        F[k][u] = set()

                        for v in self.edges[u]:
                            if v not in B[k]:    # ensure v is initialized
                                B[k][v] = set()
                            if self.dict[(u,v)] <= threshold:
                                F[k][u].add(v)
                                B[k][v].add(u) 



                preferred_edge = defaultdict(set)
                # Rule 1: dropoff(u) -> dropoff(v) for feasible (u,v)
                for u in self.dropoff_node:
                    for v in set(self.dropoff_node) & self.edges[u]:
                        preferred_edge[u].add(v)

                
                # Rule 2: pickup(u) -> dropoff(u) for all u
                for u in self.pickup_node:
                    preferred_edge[u].add(self.pickup_to_dropoff[u])
                
                # Rule 3: pickup(u) -> pickup(v) for close enough nodes
                for u in self.pickup_node:
                    for v in set(self.pickup_node) & self.edges[u] :
                        if self.dict[(u,v)] <= threshold:
                            preferred_edge[u].add(v)
                
                # Rule 4: pickup(u) -> dropoff(v) for feasible combinations
                for u in self.pickup_node:
                    for v in set(self.dropoff_node) & self.edges[u]:
                        if self.dict[(u,v)] <= threshold:
                            preferred_edge[u].add(v)

                # Rule 5: dropoff(u) -> pickup(v) for close enough nodes
                for u in self.dropoff_node:
                    for v in set(self.pickup_node) & self.edges[u]:
                        try:
                            if self.dict[(u,v)] <= threshold:
                                preferred_edge[u].add(v)
                        except:
                            print('here')
                    
                for u in self.pickup_node:
                    preferred_edge[-1].add(u)
                    preferred_edge[-1].add(u+2*self.number_of_customers)
                    preferred_edge[u+2*self.number_of_customers].add(-2)

                for v in self.dropoff_node:
                    preferred_edge[v].add(-2)
                preferred_edge[-2] = set()
                self.preferred_actions[threshold] = preferred_edge
            print('complete preferred action generation')
            #debug 
            debug = True
            if debug is True:
                this_edges = self.preferred_actions[np.inf]
                for u in this_edges:
                    if this_edges[u]!=self.edges[u]:
                        print('error edge mismatching between prefered edge and edge')
            print('check here')
    print('====end create preferred edges====')
    def _create_distance(self):
        self.distance = defaultdict()
        for u in self.pickup_node:
            self.distance[(-1,u)] = 0
            skip_node = u + 2 * self.number_of_customers
            self.distance[(-1,skip_node)]=self._slack(u)
            self.distance[(skip_node,-2)]= 0

        for v in self.dropoff_node:
            self.distance[(v,-2)] = 0
        for u in self.pickup_and_dropoff_node:
            for v in self.pickup_and_dropoff_node:
                if u!=v:
                    self.distance[(u,v)] = self._haversine_distance(u,v)


    def _default_contribution_vector(self):
        return zeros(self.number_of_customers)
        
    def _distance(self, origin, destination):
        x1, y1 = self.coordinates[origin]
        x2, y2 = self.coordinates[destination]
        return hypot(x2 - x1, y2 - y1)
    
    def _haversine_distance(self, origin, destination):
        # Early return if calculating distance to itself
        if origin == destination:
            return 0.0
        
        # Use a cache for previously calculated distances
        cache_key = (min(origin, destination), max(origin, destination))
        if hasattr(self, '_distance_cache') and cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        
        EARTH_RADIUS = 3958.8  # Radius of Earth in miles
        lat1, lon1 = map(radians, self.coordinates[origin])
        lat2, lon2 = map(radians, self.coordinates[destination])

        # Use the haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Simplified haversine calculation
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        distance = EARTH_RADIUS * c
        
        # Store in cache
        if not hasattr(self, '_distance_cache'):
            self._distance_cache = {}
        self._distance_cache[cache_key] = distance
        
        return distance
    


    def _travel_time_and_distance(self):
        """
        Vectorized full pairwise travel times & distances using numpy.
        Returns two dicts: distance[(u,v)] and travel_time[(u,v)].
        """
        nodes = list(self.pickup_and_dropoff_node)
        n = len(nodes)

        # APPROACH: Calculate as normal, then pad the matrix to allow 1-indexed access

        # 1) extract and radians‐convert coords into arrays
        lat = np.array([self.coordinates[u][0] for u in nodes])
        lon = np.array([self.coordinates[u][1] for u in nodes])
        lat_r = np.radians(lat)
        lon_r = np.radians(lon)

        # 2) pairwise deltas via broadcasting   
        dlat = lat_r[:, None] - lat_r[None, :]
        dlon = lon_r[:, None] - lon_r[None, :]

        # 3) vectorized haversine
        h1 = np.sin(dlat * 0.5)
        h2 = np.sin(dlon * 0.5)
        R = 3958.8  # miles
        a = h1**2 + np.cos(lat_r)[:,None] * np.cos(lat_r)[None,:] * (h2**2)
        dist_mat_0indexed = 2 * R * np.arcsin(np.sqrt(a))   # (n,n) matrix with 0-indexing

        # 4) compute travel times with 0-indexing
        drive_0indexed = dist_mat_0indexed / AVERAGE_SPEED
        rests_0indexed = np.floor(drive_0indexed / HOS_DRIVE_TIME)
        t_mat_0indexed = drive_0indexed + rests_0indexed * HOS_REST_TIME

        # 5) Create padded matrices for 1-indexed access
        dist_mat = np.zeros((n+1, n+1))
        dist_mat[1:n+1, 1:n+1] = dist_mat_0indexed

        drive = np.zeros((n+1, n+1))
        drive[1:n+1, 1:n+1] = drive_0indexed

        rests = np.zeros((n+1, n+1))
        rests[1:n+1, 1:n+1] = rests_0indexed

        t_mat = np.zeros((n+1, n+1))
        t_mat[1:n+1, 1:n+1] = t_mat_0indexed
        return t_mat, dist_mat, drive
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
        self.initial_res_states.add(State(-1, [self.maximum_time,0],0,self._full_resource_vec(),set(),set(),set(), 0, True, False))

        for skip_node in self.nodes[(self.number_of_customers * 2)+1:]:
            self.initial_res_states.add(State(skip_node, [self.maximum_time,0],0, self._full_resource_vec(),set(),set(),set(), 0, False, False))

        self.initial_res_states.add(State(-2, [self.maximum_time,0],0, self._empty_resource_vec(),set(),set(),set(), 0, False, True))
        
    
    def _create_null_action_info(self):
        trans_min_input = {}
        trans_term_add = {}
        trans_term_min = {}
        for res_name in self.resource_name_to_index.keys():
            trans_min_input[res_name] = 0
            trans_term_add[res_name] = 0
            trans_term_min[res_name] = np.inf
        #contribution_vector = np.zeros(len(self.rhs_vector)
        contribution_vector = np.zeros(len(self.rhs_vector)).reshape(1,-1)
        non_zero_exog_val = None
        non_zero_exog_indices = None
        cost = 0
        min_resource_vec = np.array([0,0,0,0])

        resource_consumption_vec = np.array([0,0,0,0])

        max_resource_vec = np.array([WEIGHT_CAPACITY,VOLUME_CAPACITY,self.maximum_time,MAX_COMBINED_LOADS])
        this_pickup = None
        this_dropoff = None

        self.the_single_null_action= Action(None,None,this_pickup,this_dropoff,self.number_of_customers,
                                            non_zero_exog_val,non_zero_exog_indices,cost,min_resource_vec,
                                            resource_consumption_vec,
                                            max_resource_vec, [self.maximum_time,0],0,0)
                    
    def _create_travel_time(self):
        self.travel_time = {}
        nodes = self.pickup_node+self.dropoff_node
        for n1 in nodes:
            for n2 in nodes:
                if n1 != n2:
                    self.travel_time[(n1,n2)], _ = self._travel_time_and_distance(n1,n2) + self.service_time[n1]
                else: 
                    self.travel_time[(n1,n2)] =0

    def _create_nearest_node(self,k):
        neighbors_by_distance = {
                    u: sorted(
                        [v for v in self.nodes if v != u and v not in {-1, -2} and v in self.edges[u]],
                        key=lambda v: self.actions[(u, v)][0].cost
                    )
                    for u in self.nodes if u not in {-1, -2}
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
    def _get_benefit_group(self):

        self.benefit_group = defaultdict()
        self.benefit_group_cost = defaultdict()
        for u in self.pickup_node:
            this_benefit_group = defaultdict()
            this_benefit_group_cost = defaultdict()
            for v in self.can_group[u]:
                if u != v:
                    try:
                        cost_1 = self.actions[(-1,u)][0].cost+self.actions[(u,v)][0].cost+self.actions[(v,self.pickup_to_dropoff[u])][0].cost+\
                        +self.actions[(self.pickup_to_dropoff[u],self.pickup_to_dropoff[v])][0].cost + self.actions[(self.pickup_to_dropoff[v],-2)][0].cost
                    except:
                        cost_1 = np.inf
                    try:
                        cost_2 = self.actions[(-1,u)][0].cost+self.actions[(u,v)][0].cost+self.actions[(v,self.pickup_to_dropoff[v])][0].cost+\
                        +self.actions[(self.pickup_to_dropoff[v],self.pickup_to_dropoff[u])][0].cost + self.actions[(self.pickup_to_dropoff[u],-2)][0].cost
                    except:
                        cost_2 = np.inf
                    #min_cost[(u,v)] = min(cost_1,cost_2)
                    this_benefit=  min(cost_1,cost_2)- self._slack(u)-self._slack(v) + random.uniform(-1e-6, 1e-6)
                    if this_benefit <-0.0001:
                        this_benefit_group[v] = this_benefit
                        this_benefit_group_cost[v] = this_benefit

            this_benefit_group = dict(sorted(this_benefit_group.items(), key=lambda item: item[1])) 
            self.benefit_group[u] =   this_benefit_group
            self.benefit_group_cost[u] = this_benefit_group_cost
        print('finish benefit group')
    def _define_state_update_module(self):
        # ASSIGN STATE UPDATE MODULE HERE
        self.neighbors_by_distance, self.neighbors = self._create_nearest_node(10)
        #self.plot_pickup_dropoff_locations()
        self.state_update_module = LoadAI_state_input(self.nodes, self.actions, self.weight_capacity, self.weight_demands, self.time_window_start, self.time_window_end, self.pickup_to_dropoff, self.dropoff_to_pickup, self.neighbors_by_distance, self.neighbors , self.travel_time, self.initial_resource_vector, self.resource_name_to_index, self.number_of_resources, self.problem_info)


    def plot_pickup_dropoff_with_clusters(self, threshold=0.1):
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
        
        # Find clusters of nearby points
        clusters = {}
        assigned = set()
        cluster_id = 0
        
        # Combine all nodes
        all_nodes = pickup_nodes + dropoff_nodes
        
        for node in all_nodes:
            if node in assigned:
                continue
                
            # Start a new cluster
            cluster = [node]
            assigned.add(node)
            
            # Find all points close to this node
            for other in all_nodes:
                if other in assigned or other == node:
                    continue
                    
                # Calculate Euclidean distance
                lat1, long1 = self.coordinates[node]
                lat2, long2 = self.coordinates[other]
                distance = ((lat1 - lat2) ** 2 + (long1 - long2) ** 2) ** 0.5
                
                if distance <= threshold:
                    cluster.append(other)
                    assigned.add(other)
            
            if len(cluster) > 1:
                clusters[cluster_id] = cluster
            cluster_id += 1
        
        # Add node labels with special handling for clustered nodes
        for node in pickup_nodes:
            plt.annotate(str(node), (self.coordinates[node][1], self.coordinates[node][0]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
            
        for node in dropoff_nodes:
            plt.annotate(str(node), (self.coordinates[node][1], self.coordinates[node][0]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Print clusters as tuples
        cluster_info = []
        for cluster_id, nodes in clusters.items():
            pickup_in_cluster = [n for n in nodes if n in pickup_nodes]
            dropoff_in_cluster = [n for n in nodes if n in dropoff_nodes]
            
            if pickup_in_cluster and dropoff_in_cluster:
                cluster_str = f"Cluster {cluster_id}: Pickups {tuple(pickup_in_cluster)}, Dropoffs {tuple(dropoff_in_cluster)}"
                cluster_info.append(cluster_str)
                
                # Optionally, highlight clusters on the plot
                center_lat = sum(self.coordinates[n][0] for n in nodes) / len(nodes)
                center_long = sum(self.coordinates[n][1] for n in nodes) / len(nodes)
                plt.annotate(f"Cluster {cluster_id}", (center_long, center_lat), 
                            fontsize=10, color='blue', fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.7))
        
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
        plt.show()
        # Return cluster information
        return cluster_info, plt
    def _evaluate_solution(self,excel_file_path):

        input_df = pd.read_excel(excel_file_path, sheet_name=0)
        
        # Read the output sheet (second sheet)
        output_df = pd.read_excel(excel_file_path, sheet_name=1)
        
        # Create a mapping from Shipment ID to its 1-based index in the input sheet
        shipment_id_to_index = {}
        for idx, row in input_df.iterrows():
            shipment_id = row['Shipment ID']
            shipment_id_to_index[shipment_id] = idx + 1  # 1-based indexing
        
        # Process each manifest in the output sheet
        manifest_to_indices = {}
        for _, row in output_df.iterrows():
            manifest_id = row['Manifest ID']
            
            # The Manifest ID contains hyphen-separated Shipment IDs
            shipment_ids = manifest_id.split('-')
            
            # Map each shipment ID to its index in the input sheet
            covered_indices = []
            for shipment_id in shipment_ids:
                # Convert to integer if needed
                try:
                    shipment_id = int(shipment_id)
                except ValueError:
                    pass
                
                # Find the index in our mapping
                if shipment_id in shipment_id_to_index:
                    covered_indices.append(shipment_id_to_index[shipment_id])
            
            # Store the covered indices for this manifest
            manifest_to_indices[manifest_id] = sorted(covered_indices)
        
        # Find uncovered indices
        all_indices = set(range(1, len(input_df) + 1))  # All possible 1-based indices
        covered_indices = set()
        for indices in manifest_to_indices.values():
            covered_indices.update(indices)
        
        uncovered_indices = sorted(list(all_indices - covered_indices))
        
        penalty_for_uncovered = 0
        for node in uncovered_indices:
            skipnode = node + 2 * self.number_of_customers
            penalty_for_uncovered += self.actions[(-1,skipnode)][0].cost + self.actions[(skipnode,-2)][0].cost
        #print('check here')
        return manifest_to_indices, uncovered_indices
