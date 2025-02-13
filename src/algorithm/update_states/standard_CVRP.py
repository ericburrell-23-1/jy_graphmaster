import random
from typing import List
from src.common.action import Action
from src.common.state import State
from src.algorithm.update_states.state_update_function import StateUpdateFunction


class CVRP_state_update_function(StateUpdateFunction):
    """This module is very important!!! It tells us how we will update `res_states` after pricing!!!
    
    This is definitely not complete. Just has some code to give an idea of how it will look when it is done. Needs to be fixed. Might need additional inputs.
    
    Keep in mind this module looks different for every problem type. This is just for CVRP!"""
    def __init__(self, coordinates, demands):
        super().__init__()
        pass

    def get_new_states(list_of_actions):
        return super().get_new_states()
        
    
    def _generate_beta_term(self, list_of_customer):
        idx_of_customer = {u: -1 for u in self.nodes if u not in {-1,-2}}
        for idx in range(len(list_of_customer)):
            idx_of_customer[list_of_customer[idx]] = idx
        customer_not_in_route = list(set(self.nodes)-set(list_of_customer)-{-1,-2})
        for customer in customer_not_in_route:
            for customer2 in self.sorted_neighbors[customer]:
                if customer2 in list_of_customer:
                    idx_of_customer[customer] = idx_of_customer[customer2] + \
                        round(random.random(), 5)*0.01
                    break
        beta = sorted(idx_of_customer, key=lambda k: idx_of_customer[k])

        return beta

    def _generate_state_based_on_beta(self, beta):
        """This needs to be reviewed but basically this is the function that should be called in `get_new_states`."""
        myState = []
        dem_list = []
        for idx in range(len(beta)):
            customer = beta[idx]
            myCanVisit = {
                f'can_visit: {u}': 0 if beta.index(u) < beta.index(customer) else 1
                for u in self.nodes if u not in {-1, -2}
            }
            if idx>0:
                dem_list.append(self.problem_data['demands'][beta[idx-1]])
            minimum_dem_remain = self.problem_data['capacity'] - self.problem_data['demands'][customer]
            poss_demand_used = self.get_unique_value_from_list(dem_list,minimum_dem_remain)
            poss_demand_used.append(0)
            for d in poss_demand_used:
                resource_vector = myCanVisit.copy()
                resource_vector['cap_remain'] = self.problem_data['capacity']-d
                this_state = State(customer, resource_vector, [], [])
                myState.append(this_state)

        source_state_resource_vector = {
            f'can_visit: {u}': 1 for u in self.nodes if u not in {-1, -2}}
        source_state_resource_vector['cap_remain'] = self.problem_data['capacity']
        source_state = State(-1, source_state_resource_vector, [], [])
        sink_state_resource_vector = {
            f'can_visit: {u}': 0 for u in self.nodes if u not in {-1, -2}}
        sink_state_resource_vector['cap_remain'] = 0
        sink_state = State(-2, sink_state_resource_vector, [], [])
        myState.append(source_state)
        myState.append(sink_state)
        return set(myState)

    def get_states_from_action_list(self, action_list: List[Action]):
        """
        Returns a list of states given an action_list.
        """
        if not action_list:
            return []
        states_list = []
        current_resources = self.initial_resource_state.copy()
        pred_state = State(action_list[0].origin_node, current_resources, [], [])
        states_list.append(pred_state)
        for action in action_list:
            new_resource_vector = action.compute_output_resource_vector(current_resources)
            if new_resource_vector is None:
                print(f"Invalid resource transition from {action.origin_node} to {action.destination_node}")
                break
            new_state = State(action.destination_node, new_resource_vector, [], [pred_state])
            pred_state.successor_states.append(new_state)
            states_list.append(new_state)
            current_resources = new_resource_vector.copy()
            pred_state = new_state
        return states_list
    