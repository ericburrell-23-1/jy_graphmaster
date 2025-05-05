from random import randint
import numpy as np
class State:
    def __init__(self, node:int, time_window,service_time,  state_vec,picked_up, dropped_off,must_dropoff, l_id: int, is_source: bool, is_sink: bool):
        if node == None:
            input('error here for state')
        self.node = node
        self.state_vec = state_vec
        self.picked_up = picked_up
        self.dropped_off = dropped_off
        self.must_drop_off = must_dropoff
        self.l_id=l_id #id for the l in Omega_R.  we can give each graph its own source and sink that does not matter
        self.is_source=is_source #indicates if source
        self.is_sink=is_sink#indicates if sink
        self.time_window = time_window
        self.service_time = service_time
        
        self._check_state()
        self.state_id= hash((self.node,self.is_sink,self.is_source,self.l_id,tuple(self.state_vec),tuple(picked_up),tuple(dropped_off)))

    def _check_state(self):
        debug_here = False
        if debug_here== True:
            if self.state_vec[5]>840 or self.state_vec[4]>660:
                input('error here for state check')
            if np.any(self.state_vec<0):
                print('statevec')
                print(self.state_vec)
                print('node')
                print(self.node)
                input('error:some vec less than 0')
            
        if self.state_vec[4]>self.state_vec[5]:
            self.state_vec[4] = self.state_vec[5]
         

    def __eq__(self, other: 'State') -> bool:
        if other is None:
            return False
        return self.state_id == other.state_id

    def __hash__(self) -> int:
       """
       Provides a hash so that State objects can be used in sets or as dictionary keys.
       We hash by the node and the contents of res_vec.
       """
       return self.state_id

    def this_state_dominates_input_state(self, other_state):
        """
        Determines if this state dominates the input `other_state`.
        Also determines if a tie occurs.
        """
        does_dom = False
        does_equal = False

        # Ensure both states belong to the same node before comparison
        if other_state.node != self.node:
            return [False, False]

        # Convert sparse vectors to dense NumPy arrays to align indices
        vec1_dense = self.state_vec
        vec2_dense = other_state.state_vec
        
        picked_up_2 = other_state.picked_up
        dropped_off_2 = other_state.dropped_off
        # Compute element-wise difference

        res_vec_diff = vec1_dense - vec2_dense
    

        # Compute min and sum values
        min_value = res_vec_diff.min()  # Minimum difference
    
        sum_value = np.abs(res_vec_diff).sum()  # Absolute sum of differences

        # Domination condition
        if min_value >= 0 and sum_value > 0 and picked_up_2==self.picked_up and dropped_off_2.issubset(self.dropped_off):
            does_dom = True

        # Equality check
        if np.array_equal(vec1_dense, vec2_dense) and self.picked_up==picked_up_2 and self.dropped_off==dropped_off_2:
            does_equal = True

        return [does_dom, does_equal]

    def pretty_print_state(self):
        print('state description')
        print('l_id:  '+str(self.l_id))
        print('node:  '+str(self.node))
        print('stateVec:   '+str(self.state_vec.toarray()))
        print('state_id:  '+str(self.state_id))


    def equals(self,secondary_state):
        flag = self.state_id == secondary_state.state_id
        return flag

    def is_source(self):
        return self.node == -1
    def is_sink(self):
        return self.node == -2
    
    def service(self):
        if self.node == -2:
            return [self]
        REST_DURATION = 660  # Define the rest duration as a constant
        MAX_WORK_AFTER_REST = 840
        earlist_service_start =  min(self.state_vec[2],self.time_window[0])
        wait_time = self.state_vec[2]-earlist_service_start
        depart_states = []
        if self.state_vec[5]>self.service_time + wait_time:
            this_state_vec = self.state_vec.copy()
            this_state_vec[2] = earlist_service_start-self.service_time
            if this_state_vec[2]>0:
                this_state_vec[5] -= (self.service_time+wait_time)
                # if np.any(this_state_vec<0):
                #     input('state_no_rest service generate negative resource')
                state_no_rest = State(self.node,self.time_window,self.service_time, this_state_vec,self.picked_up,self.dropped_off,
                        self.must_drop_off,self.l_id,self.is_source,self.is_sink)
                depart_states.append(state_no_rest)

        earlist_service_start = min(self.state_vec[2]-REST_DURATION,self.time_window[0]   )

        
        this_state_vec = self.state_vec.copy()
        this_state_vec[2] = earlist_service_start - self.service_time
        this_state_vec[4] = REST_DURATION
        this_state_vec[5] = MAX_WORK_AFTER_REST-self.service_time
        if this_state_vec[2]>0:
            # if np.any(this_state_vec<0):
            #     input('state_rest_service service generate negative resource')
            state_rest_service = State(self.node,self.time_window,self.service_time,this_state_vec,self.picked_up,self.dropped_off,
                    self.must_drop_off,self.l_id,self.is_source,self.is_sink)
            depart_states.append(state_rest_service)

        return depart_states
        
