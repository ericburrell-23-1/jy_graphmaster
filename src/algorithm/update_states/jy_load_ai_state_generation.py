
from itertools import permutations
import numpy as np
from scipy.sparse import csr_matrix
from src.common.state import State
#jy_load_ai_state_generation
from src.algorithm.jy_slow_pricing import SortedObjectList
class jy_make_load_ai_states:

    def set_depth_used_by_action():
        self.DepthUsed=dict()

        Q=self.SLAI
        for a in Q.actions:
            node_destination=a.node_head
            if node_destination in Q.pickup_nodes:
                self.DepthUsed[a]=1
            else:
                self.DepthUsed[a]=0
    def __init__(self,shawn_LoadAI_state_input,list_of_action_in_col_ordered,list_of_states_in_col_ordered,jy_options):
        print('hello world')
        self.SLAI=shawn_LoadAI_state_input
        self.list_of_action_in_col_ordered=list_of_action_in_col_ordered
        self.list_of_states_in_col_ordered=list_of_states_in_col_ordered
        self.shawn_LoadAI_state_input = shawn_LoadAI_state_input
        self.jy_options=jy_options
        self.node_min_term_vec = shawn_LoadAI_state_input['node_min_term_vec']
        self.MaxDepth=self.jy_options['max_pickups_in_a_route']#S.max_depth
        self.option_do_min_term=True
        self.set_depth_used_by_action()
        self.update_action_subset_given_col()
        self.updeate_action_from_nodes_subset()
        self.init_states_project_and_depth
        #assign depth used to all actions;  Feel free to remove this later
        
        self.gen_all_states_naive()
    
    def init_states_project_and_depth(self):
        self.my_sorted=SortedObjectList()
 
        self.states_can_expand=[]
        if self.option_do_min_term==True:
            tmp=[]
            for s in self.list_of_states_in_col_ordered:
                s2=self.apply_node_min_term(s)
                tmp.append(s2)
                self.states_can_expand.append(s2)
                self.my_sorted.insert(s2,self.MaxDepth)
                
            self.list_of_states_in_col_ordered=tmp
        self.State2Depth=dict()
        for s in self.states_can_expand:
            self.State2Depth[s]=self.MaxDepth
        
    
    def return_solution(self):
        return self.all_states


    def confirm_if_state_possible(self,candid_state):
        print('hello world')
        pickup_node = self.shawn_LoadAI_state_input['pick up node']
        dropoff_node = self.shawn_LoadAI_state_input['drop off node']
        s2 = candid_state
        if s2.node in pickup_node:
            if 0>0:
                pick_up_vec = s2.state_vec[0,4:4+len(pickup_node)]
                drop_off_vec = s2.state_vec[0,4+len(pickup_node):]
                dense_array = pick_up_vec.toarray()[0]
                zero_indices = np.where(dense_array == 0)[0]
                drop_off_node_need_to_visit = []
    
                for n in zero_indices:
                    if drop_off_vec[0,n] == 0:
                        drop_off_node_need_to_visit.append(n+1+len(pickup_node))
            else:
                drop_off_node_need_to_visit = []
                drop_off_vec = s2.state_vec[0,4+len(pickup_node):]
                dense_array = drop_off_vec.toarray()[0]
                zero_indices = np.where(dense_array == 0)[0]
                for n in zero_indices:
                    drop_off_node_need_to_visit.append(n+1+len(dropoff_node))
            if len(drop_off_node_need_to_visit) >0:
                permutation_of_drop_off_node = list(permutations(drop_off_node_need_to_visit))
            
                valid = False
    
                for list_of_node in permutation_of_drop_off_node:
                    pre_node = s2.node
                    pre_state = s2
                
                    go_next_loop = False
                    for this_node in list_of_node:
                        a = self.actions[(pre_node,this_node)][0]
                        new_state = a.get_head_state(pre_state,pre_state.l_id)
                        if new_state == None:
                            go_next_loop = True
                            break
                        pre_node = this_node
                        pre_state = new_state
                    if go_next_loop:
                        continue
                    if (pre_node,-2) not in self.actions:
                        input('error here')
                    a = self.actions[(pre_node,-2)][0]
                
                    new_state = a.get_head_state(pre_state,pre_state.l_id)
                    if new_state != None:
                        valid = True
                        break
            else:
                valid = True
            if valid == True:
                return True
            else:
                return False
        elif s2.node in dropoff_node:
            if 0>0:
                pick_up_vec = s2.state_vec[0,4:4+len(pickup_node)]
                drop_off_vec = s2.state_vec[0,4+len(pickup_node):]
                nonzero_indices = pick_up_vec.indices
                nonzero_values = pick_up_vec.data
                zero_indices = nonzero_indices[np.isclose(nonzero_values, 0)]
                drop_off_node_need_to_visit = []
                for n in zero_indices: # remove dropoff of current node
                    if drop_off_vec[0,n] == 0 and n+1+len(pickup_node) != s2.node:
                        drop_off_node_need_to_visit.append(n+1+len(pickup_node))
            else:
                drop_off_node_need_to_visit = []
                drop_off_vec = s2.state_vec[0,4+len(pickup_node):]
                dense_array = drop_off_vec.toarray()[0]
                zero_indices = np.where(dense_array == 0)[0]
                for n in zero_indices:
                    if n+1+len(pickup_node) != s2.node:
                        drop_off_node_need_to_visit.append(n+1+len(dropoff_node))
            if len(drop_off_node_need_to_visit) >0:
                permutation_of_drop_off_node = list(permutations(drop_off_node_need_to_visit))
            
                valid = False
    
                for list_of_node in permutation_of_drop_off_node:
                    pre_node = s2.node
                    pre_state = s2
                    go_next_loop = False
                    for this_node in list_of_node:
                        a = self.actions[(pre_node,this_node)][0]
                        new_state = a.get_head_state(pre_state,pre_state.l_id)
                        if new_state == None:
                            go_next_loop= True
                            break
                        pre_node = this_node
                        pre_state = new_state
                    if go_next_loop:
                        continue
                    a = self.actions[(pre_node,-2)][0]
                    new_state = a.get_head_state(pre_state,pre_state.l_id)
                    if new_state != None:
                        valid = True
                        break
            else:
                a = self.actions[(s2.node,-2)][0]
                new_state = a.get_head_state(s2,s2.l_id)
                if new_state != None:
                    valid = True
            if valid == True:
    
                return True
            else:
                return False
        else:
            input('state is not pickup or drop off node')
        #checkign my permuations
        #self.SLAI is teh shawn state generation structure
        #if its valdi then return true otehrwise return fasle
    
    def apply_node_min_term(self,in_state):
        print('hellow orld')
        s = in_state
        new_state_vec = self.elementwise_min_csr(s.state_vec, self.nodes_min_term_vec[s.node])
        if np.sum(np.abs(new_state_vec-s.state_vec)) > .00001:
            s2 = State(s.node, new_state_vec, s.l_id, s.is_source, s.is_sink)
            return s2
        else:
            return s   

    def jy_can_expand(self,s_origin,my_action,option_do_min_term):
        #return false if takign this action from theis state produces an ifeasible action
        #possibilites
        #possibility s_deestination is none
        
        #the my_action.node_head (meaning destination ) is a dropoff  and the MustAvoidDropOff is not active for it OR current location is not the associated pickup
             #rmember to include if you are at the pickup for htat customer.  thats the speical case

        #possibility confirm_if_state_possible returns false

        #possibility:  there would be more than maxPickupsInstate pickups overall; 
            #remember you have a reource for this. so this should be caught by get head state but do check
            #rmember to include if you are at the pickup for htat customer
        #possibility:  if option_do_min_term is false then you can count exactly how many pickups you have 
            #make sure to include the pickup for the current node if it is a pickup
        
        
        #possibility: the number of dropoffs required exceeds depth remaining:
            #make sure to model the current node thing.  
        #return true otherwise
        print('hello world')
        return True

    def update_action_subset_given_col(self):
        Q=self.SLAI
        self.Action_subset=Q.action_reasonable.copy()
        for i_ind in range(0,len(self.list_of_states_in_col_ordered)):
            for j_ind in range(i_ind+1,len(self.list_of_states_in_col_ordered)):
                s1=    self.list_of_states_in_col_ordered[i_ind]
                s2=    self.list_of_states_in_col_ordered[j_ind]
                my_actions_n1_n2=Q.actions[s1.node,s2.node]
                self.Action_subset=self.Action_subset.union(my_actions_n1_n2)
    def updeate_action_from_nodes_subset(self):
        Q=self.SLAI
        self.actions_from_node_subset=dict()
        #self.actions_from_node_subset_pickup=dict()
        self.actions_from_node_subset_dest_dropoff=dict()
        #self.actions_from_node_subset_sink=dict()
        self.actions_from_node_subset_MINUS_dest_dropoff=dict()
        for n in Q.nodes:
            self.actions_from_node_subset[n]=[]
        for a in self.Action_subset:
            my_origin=a.node_tail
            my_destination=a.node_head
            do_add=False
            if self.option_do_min_term:
                #shawn 
                if np.min(a.origin.NodeMinTermVec.toarray()- a.minTermInput.toarray())>=-0.0001:
                    do_add=True
                    self.actions_from_node_subset[my_origin].append(a)
            else:
                do_add=True
            if do_add==True:
                self.actions_from_node_subset[my_origin].append(a)
                if my_destination in Q.dropoff_nodes:
                    self.actions_from_node_subset_dest_dropoff[my_origin].append(a)
                else:
                    self.actions_from_node_subset_MINUS_dest_dropoff[my_origin].append(a)

    def expand_state_given_action(self,s,my_act,orig_depth_s):

        can_expand=self.jy_can_expand(my_act,s,self.option_do_min_term)
        if can_expand==False:
            return False,None,None
        #\State  $s_2\leftarrow GetHeadState(s\rightarrow a)$
        my_head=my_act.get_head_state(s)
        my_new_depth=orig_depth_s-self.DepthUsed[my_act]
        #\If {$s_2== None$}
        did_make_term=False
        debug_on=True
        #\State Continue
        did_make_term=True
        #\EndIf
        if debug_on:
            if False==self.confirm_if_state_possible(my_head):
                input('error here 1')
        #\State $s_2.stateVec\leftarrow ElementwiseMin(s_2.stateVec,s2.node.NodeMinTermVec)$

        my_head=self.apply_node_min_term(my_head)
        if debug_on:
            if False==self.confirm_if_state_possible(my_head):
                input('error here 2')
        return did_make_term,my_head,my_new_depth

    def get_must_drop_off_including_current(s):
        Q=self.SLAI
        
        must_drop_off=[]
       
        
        may_avoid_dropoff=s.state_vec[4+self.num_pickups:]
        must_dropoff=np.nonzero(may_avoid_dropoff<0.5)
        must_dropoff=may_avoid_dropoff_list+self.num_pickups
        if s.node in Q.pickup_nodes:
            must_dropoff.append(s.node+self.num_pickups)
        return must_dropoff
    def actions_from_node_subset(self,s):

        actions_use=self.actions_from_node_subset_MINUS_dest_dropoff[s.node].copy()
        must_dropoff=self.get_must_drop_off_including_current(s)
        for n in must_dropoff:
            my_act_list=self.Q.node_2_actions[s.node,n.node]
            for my_act in my_act_list:
                actions_use.append(my_act)
        return actions_use
    def gen_all_states_naive(self):

        Q=self.SLAI
        
        while len(self.states_can_expand>0):
            #State $s\leftarrow \mbox{arg} \max_{s\in StatesCanExpand}State2Depth(s)$
            #Pop(s)$ from $StatesCanExpand$.  Always select to expand the term with $State2Depth$ 

            s=self.my_sorted.pop_max()
            orig_depth_s=self.State2Depth[s]

            actions_use=self.actions_from_node_subset(s)
            for my_act in actions_use:
                [did_make_new_state,my_head,my_new_depth]=self.expand_state_given_action(s,my_act,orig_depth_s)
               
                if did_make_new_state==True  and my_head not in self.State2Depth and my_new_depth>-0.5:
                    self.State2Depth[my_head]=my_new_depth
                    self.states_can_expand.insert(my_head,my_new_depth)

    def elementwise_min_csr(self,vec1: csr_matrix, vec2: csr_matrix) -> csr_matrix:
        """
        Compute the elementwise minimum of two CSR matrices.
        """
        vec2 = vec2.reshape(1, -1)
 
        if vec1.shape != vec2.shape:
            raise ValueError(f"Matrices have incompatible shapes: {vec1.shape} vs {vec2.shape}")
       
        # Convert to COO format for easier manipulation
        cx1 = vec1.tocoo()
        cx2 = vec2.tocoo()
       
        # Create dictionaries for non-zero values
        dict1 = {(i, j): v for i, j, v in zip(cx1.row, cx1.col, cx1.data)}
        dict2 = {(i, j): v for i, j, v in zip(cx2.row, cx2.col, cx2.data)}
       
        # Combine keys
        all_keys = set(dict1.keys()).union(set(dict2.keys()))
       
        # Create new data for the minimum values
        rows, cols, data = [], [], []
        for i, j in all_keys:
            # Get values, with 0 as default (not infinity) for missing keys
            val1 = dict1.get((i, j), 0)
            val2 = dict2.get((i, j), 0)
            min_val = min(val1, val2)
           
            # Only include non-zero values in the result
            if min_val != 0:
                rows.append(i)
                cols.append(j)
                data.append(min_val)
       
        # Create a new CSR matrix
        return csr_matrix((data, (rows, cols)), shape=vec1.shape)
