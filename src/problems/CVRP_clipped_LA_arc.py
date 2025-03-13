from collections import ChainMap
from itertools import chain, combinations
from src.problems.optimization_problem import OptimizationProblem
from src.common.state import State
from src.common.action import Action
from src.algorithm.update_states.standard_CVRP import CVRP_state_update_function
import numpy as np
from numpy import zeros, ones
from math import hypot
from scipy.sparse import csr_matrix
from src.common.helper import Helper
 
class CVRP_Clipped_LA_arc(OptimizationProblem):
 

    def get_valid_subsets(self,neighbors, u_p):
        """Returns all non-empty subsets of neighbors that satisfy the capacity constraint."""
        valid_subsets = []
        for r in range(1, len(neighbors) + 1):
            for subset in combinations(neighbors, r):
                total_demand = self.demands(u_p) + sum(self.demands(w) for w in subset)
                if total_demand <= self.capacityRemaining:
                    valid_subsets.append(set(subset))
        return valid_subsets

    def get_all_valid_node_tuples(self):
        """Generates tuples (u_p, N_p, v_p) for each node u in self.nodes, ensuring demand constraint.
        
        - u_p is a node.
        - N_p is a subset of the neighbors of u_p.
        - v_p is an element of N_p.
        - Restriction: self.capacityRemaining ≥ sum(self.demand(w) for w in {u_p} ∪ N_p ∪ {v_p}).
        - Special case: If N_p is empty, then u_p = v_p.
        """
        
        

        all_tuples = []
        
        for u in self.nodes:
            neighbors = self.neighbors_by_distance.get(u, [])

            # Generate valid subsets of neighbors based on capacity constraint
            for N_p in self.get_valid_subsets(neighbors, u):
                for v_p in N_p:
                    total_demand = self.demands[u] + sum(self.demands[w] for w in N_p) + self.demands[v_p]
                    if total_demand <= self.capacityRemaining:
                        all_tuples.append((u, N_p, v_p))

            # Special case: u_p = v_p when N_p is empty (single node case)
            #if self.demands[u] <= self.capacity:
            all_tuples.append((u, set(), u))
    
        self.all_LA_tuples_sorted = sorted(
            [tup for tup in all_tuples if tup[0] != tup[2]],  # Filter where u_p ≠ v_p
            key=lambda tup: len(tup[1])  # Sort by |N_p|
        )

        self.all_clipped_la_arcs=[]
        self.tup_2_cost=dict()
        self.tup_2_ordering=dict()
        for my_tup in self.all_LA_tuples_sorted:
            u_p=my_tup[0]
            N_p=my_tup[1]
            v_p=my_tup[2]
            cost=0
            this_ordering=[]
            if len(N_p==1):
                cost=self._distance(u_p, v_p)
                this_ordering=[u_p, v_p]
            else:
                cost=np.inf
                for w in N_p:
                    if w!=v_p:
                        u2_p=w
                        N2_p=my_tup[1].setdiff1d(w)
                        v2_p=my_tup[2]
                        my_candid_tup=tuple([u2_p,N2_p,v2_p])
                        this_cost=self._distance(u_p, w)+self.tup_2_cost[my_candid_tup]
                        if this_cost<cost:
                            cost=this_cost
                            this_ordering=[u_p]+self.tup_2_ordering[my_candid_tup]
            
            self.tup_2_cost[my_tup]=cost
            self.tup_2_ordering[my_tup]=this_ordering
    
        self._closest_k_neighbors()
        distant_pairs = [
            (u, v) 
            for u in self.nodes 
            for v in self.nodes
            if v != u and v not in self.node_2_la_neigh[u]  # Exclude neighbors and self-loops
            and self.demands[u] + self.demands[v] <= self.capacity  # Capacity constraint
            and v!=-1
            and u!=-2
        ]
        for (u,v) in distant_pairs:
            if u==-1 and v==-2:
                continue
            N_p=frozenset([v])
            if v!=-2:
                N_p=frozenset([])
            my_tup=tuple([u,N_p,v])
            self.tup_2_cost[my_tup]=self._distance(u, v)


            self.tup_2_ordering[my_tup]=[u,v]
        
        self.tup_2_dem_used=dict()
        self.tup_2_cover=dict()
        for my_tup in self.tup_2_cost:
            u_p=my_tup[0]
            N_p=my_tup[1]
            v_p=my_tup[2]
            cust_cover=set([])
            tot_dem=0
            if u_p!=-1:
                tot_dem=tot_dem+self.demands[u_p]
                cust_cover.add(u_p)
            for w in N_p:
                if w!=v_p:
                    tot_dem=tot_dem+self.demands[w]
                    cust_cover.add(w)
            self.tup_2_cover[my_tup]=cust_cover
            self.tup_2_dem_used[my_tup]=tot_dem

    def __init__(self, problem_instance_file_name: str, file_type: str = "Standard_VRP"):
        """Defines all aspects of a CVRP problem needed before calling the `solve` method."""
        self.neighbors_by_distance = {}
        self.num_LA_neigh=10
        super().__init__(problem_instance_file_name, file_type)
        
        self._generate_neighbors()
        self.LA_generate_LA_neighbors()
        self._create_null_action_info()
 
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
                    input('----')
 
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
                    input('----')
        self.capacity = capacity
        self.demands = demands
        self.coordinates = coordinates
    
 
    def _build_problem_model(self):
        num_customers = len(self.demands) - 2
        self.nodes = [-1]
        self.initial_resource_dict = {"cap_remain": self.capacity}
        self.rhs_vector = ones(num_customers)
        self.actions = {}
        self.OLD_actions = {}
        idx = 0
        self.constraint_name_to_index = {}
        for node in self.demands.keys():
            if node not in {-1,-2}:
                self.nodes.append(node)
                self.initial_resource_dict[f'can_visit: {node}'] = 1
        print('self.initial_resource_dict')
        print(self.initial_resource_dict)
        input('-==-')
        self.nodes.append(-2)
        self.number_of_resources = num_customers + 1 # number of customer + source + sink + capremain
        full_resource_dict = np.ones(self.number_of_resources)
        full_resource_dict[0] = self.capacity
        full_resource_vec = csr_matrix(full_resource_dict.reshape(1, -1))
        empty_resource_dict = np.zeros(self.number_of_resources)
        empty_resource_vec = csr_matrix(empty_resource_dict.reshape(1, -1))
        

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

        
        self.default_exog_name_to_coeff_dict = {}
        for node in self.nodes:
             self.default_exog_name_to_coeff_dict[("Cover", node)] = 0
        
        idx = 0
        partial_max_resource_dict = {"cap_remain":self.capacity}
        for node in self.nodes:
            if node not in {-1,-2}:
                partial_max_resource_dict[f'can_visit: {node}'] =1

        idx=0
        for origin_node in self.nodes:
            if origin_node > 0:
                # DEFINE COVERAGE CONSTRAINT RHS
                self.constraint_name_to_index[str(("Cover", origin_node))] = idx
                self.rhs_constraint_name_to_index[str(("Cover", origin_node))] = idx
                self.rhs_index_to_constraint_name[idx] = str(("Cover", origin_node))
                idx += 1
        idx = 1
        for u in self.nodes:
            if u in (-1,-2):
                continue
            self.default_min_resource_dict[f'can_visit: {u}'] = 0
            self.default_max_resource_dict[f'can_visit: {u}'] = 1
            self.default_resource_consumption_dict[f'can_visit: {u}'] = 0
            self.resource_name_to_index[f'can_visit: {u}'] = idx
            idx +=1
        self.get_all_valid_node_tuples()

        my_tuples=self.tup_2_cover.keys()
        print('my_tuples')
        print(my_tuples)
        input('go')
        for origin_node in self.nodes:
            for destination_node in self.nodes:
                if origin_node==-2 or destination_node==-1 or origin_node==destination_node:
                    continue
                self.actions[origin_node,destination_node]=[]
        for my_tup in my_tuples:
            u_p=my_tup[0]
            N_p=my_tup[1]
            v_p=my_tup[2]
            if u_p==-2 or v_p==-1 or u_p==v_p:
                print('my_tup')
                print(my_tup)
                input('error here')
            origin_node=u_p
            destination_node=v_p
            cost=self.tup_2_cost[my_tup]
            this_dem=self.tup_2_dem_used[my_tup]
            this_dem_extra=this_dem
            if v_p!=-2:
                this_dem_extra=this_dem_extra+self.demands[v_p]
            this_cust_cover=self.tup_2_cover[my_tup]
            ####
            contribution_vector=zeros(num_customers)
            for w in this_cust_cover:
                contribution_vector[self.constraint_name_to_index[str(("Cover", w))]] = 1
                partial_resource_consumption_dict[f'can_visit: {w}'] =-1
            #if v_p!=-2:
            #    partial_min_resource_dict
            partial_min_resource_dict = {"cap_remain":this_dem_extra}
            partial_resource_consumption_dict = {"cap_remain": -this_dem}
            
            
            if v_p!=-2:
                partial_min_resource_dict[f'can_visit: {destination_node}']= 1
            

            ####
            trans_min_input = ChainMap(partial_min_resource_dict, self.default_min_resource_dict)
            trans_term_min = ChainMap(partial_max_resource_dict, self.default_max_resource_dict)
            trans_term_add = ChainMap(partial_resource_consumption_dict, self.default_resource_consumption_dict)
            

            _,min_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_min_resource_dict)     
            _,resource_consumption_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_resource_consumption_dict)     
            _,max_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_max_resource_dict)     
            indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,partial_max_resource_dict)
            my_action = Action(trans_min_input,trans_term_add,trans_term_min,destination_node,origin_node,contribution_vector,cost,min_resource_vec,resource_consumption_vec,indices_apply_min_to,max_resource_vec,full_resource_dict,empty_resource_vec)
            self.actions[origin_node, destination_node].append(my_action)
                    
        
        for origin_node in self.nodes:
            
            for destination_node in self.nodes:
                if origin_node == destination_node or origin_node==-2 or destination_node == -1:
                    continue
                if origin_node == -1 and destination_node == -2:
                    continue
                cost = self._distance(origin_node, destination_node)
                #print(origin_node,destination_node,cost)
                contribution_vector = zeros(num_customers)
                if origin_node > 0:
                    contribution_vector[self.constraint_name_to_index[str(("Cover", origin_node))]] = 1
                partial_min_resource_dict = {"cap_remain": self.demands[origin_node] + self.demands[destination_node]}
                partial_resource_consumption_dict = {"cap_remain": -self.demands[origin_node]}
                
                if origin_node != -1:
                    partial_resource_consumption_dict[f'can_visit: {origin_node}'] =-1
                if destination_node!=-2:
                    partial_min_resource_dict[f'can_visit: {destination_node}']= 1
                
                trans_min_input = ChainMap(partial_min_resource_dict, self.default_min_resource_dict)
                trans_term_min = ChainMap(partial_max_resource_dict, self.default_max_resource_dict)
                trans_term_add = ChainMap(partial_resource_consumption_dict, self.default_resource_consumption_dict)
                

                _,min_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_min_resource_dict)     
                _,resource_consumption_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_resource_consumption_dict)     
                _,max_resource_vec = Helper.dict_2_vec(self.resource_name_to_index,self.number_of_resources,partial_max_resource_dict)     
                indices_apply_min_to=Helper.partial_map_2_indices_applied(self.resource_name_to_index,partial_max_resource_dict)
                action = Action(trans_min_input,trans_term_add,trans_term_min,destination_node,origin_node,contribution_vector,cost,min_resource_vec,resource_consumption_vec,indices_apply_min_to,max_resource_vec,full_resource_dict,empty_resource_vec)
                self.OLD_actions[origin_node, destination_node] = [action]
                
 
    def _create_null_action_info(self):
        full_resource_dict = np.ones(self.number_of_resources)
        full_resource_dict[0] = self.capacity
        full_resource_vec = csr_matrix(full_resource_dict.reshape(1, -1))
        empty_resource_dict = np.zeros(self.number_of_resources)
        empty_resource_vec = csr_matrix(empty_resource_dict.reshape(1, -1))
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
        

    def _define_state_update_module(self):
        """This is where we define how res_states is updated after pricing. We are using the `standard_CVRP` module for this definition."""
        self.state_update_module = CVRP_state_update_function(self.nodes, self.actions,self.capacity, self.demands, self.neighbors_by_distance, self.initial_resource_vector,self.resource_name_to_index,self.number_of_resources)
    
    def _generate_neighbors(self):
        self.neighbors_by_distance = {
            u: sorted(
                [v for v in self.nodes if v != u],
                key=lambda v: self._distance(u, v)
            )
            for u in self.nodes
        }

        
        

    def _closest_k_neighbors(self):
        """Computes the K nearest neighbors for each node, ensuring:
        - Nodes `-1` and `-2` have empty neighborhoods.
        - Customers `v` are excluded from `u`'s neighborhood if `self.demand(u) + self.demand(v) > self.capacity`.
        """
        
        self.node_2_la_neigh={
            u: [] if u in {-1, -2} else sorted(
                [
                    v for v in self.nodes 
                    if v != u and v not in {-1, -2}  # Exclude -1 and -2
                    and self.demands[u] + self.demands[v] <= self.capacity  # Exclude if demand exceeds capacity
                ],
                key=lambda v: self._distance(u, v)
            )[:self.num_LA_neigh]  # Keep only the K nearest neighbors
            for u in self.nodes
        }

 
    def _distance(self, origin, destination):
        """Helper function to get costs of actions easily."""
        x1, y1 = self.coordinates[origin]
        x2, y2 = self.coordinates[destination]
 
        return hypot(x2 - x1, y2 - y1)
 