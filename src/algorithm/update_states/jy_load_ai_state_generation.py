
from itertools import permutations
import numpy as np
from scipy.sparse import csr_matrix
from src.common.state import State
#jy_load_ai_state_generation
from src.algorithm.jy_slow_pricing import SortedObjectList
class jy_make_load_ai_states:

    def set_depth_used_by_action(self):
        self.DepthUsed=dict()

        Q=self.SLAI
        for (n1,n2) in Q.actions:
            for a  in Q.actions[n1,n2]:
                node_destination=a.node_head
                if node_destination in Q.pickup_node:
                    self.DepthUsed[a]=1
                else:
                    self.DepthUsed[a]=0
    def debug_gen_states_from_actions(self):
        cur_state=self.list_of_states_in_col_ordered[0]
        for ai in range(0,len(self.list_of_action_in_col_ordered)):
            my_act=self.list_of_action_in_col_ordered[ai]
            cur_state=my_act.get_head_state(cur_state,cur_state.l_id)
            target_state=self.list_of_states_in_col_ordered[ai+1]
            if False==cur_state.equals(target_state):
                print('target_state.pretty_print_state()')
                target_state.pretty_print_state()
                print('cur_state.pretty_print_state()')
                
                cur_state.pretty_print_state()
                print('ai ')
                print(ai)
                input('error here ')
    def __init__(self,shawn_LoadAI_state_input,list_of_states_in_col_ordered,list_of_action_in_col_ordered,jy_options):
        #print('hello world')
        self.all_states=[]

        self.SLAI=shawn_LoadAI_state_input
        self.list_of_action_in_col_ordered=list_of_action_in_col_ordered
        self.list_of_states_in_col_ordered=list_of_states_in_col_ordered
        self.actions = shawn_LoadAI_state_input.actions
        self.debug_gen_states_from_actions()
        self.pickup_node = shawn_LoadAI_state_input.pickup_node
        self.dropoff_node = shawn_LoadAI_state_input.dropoff_node
        self.resource_name_to_index = shawn_LoadAI_state_input.resource_name_to_index
        self.jy_options=jy_options
        self.node_min_term_vec = shawn_LoadAI_state_input.node_min_vec_dict
        self.MaxDepth=self.jy_options['max_pickups_in_a_route']#S.max_depth
        self.option_do_min_term=False
        self.use_all_actions=True
        self.set_depth_used_by_action()
        self.update_action_subset_given_col()
        self.updeate_action_from_nodes_subset()
        self.init_states_project_and_depth()
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
                self.all_states.append(s2)
            self.list_of_states_in_col_ordered=tmp
        else:
            for s in self.list_of_states_in_col_ordered:
                self.states_can_expand.append(s)
                self.my_sorted.insert(s,self.MaxDepth)
                self.all_states.append(s)
        self.State2Depth=dict()
        for s in self.states_can_expand:
            self.State2Depth[s]=self.MaxDepth
        
    
    def return_solution(self):
        print(' in returing self.all_states')
        print(self.all_states)
        print('returning solution')


        for s in self.list_of_states_in_col_ordered:
            if s not in self.all_states:
                input('error here not found')

        return [self.all_states,self.list_of_states_in_col_ordered]


    def confirm_if_state_possible(self,candid_state):
        print('hello world')
        print('candid_state')
        candid_state.pretty_print_state()
        pickup_node = self.pickup_node
        dropoff_node = self.dropoff_node
        s2 = candid_state
        if candid_state.node in {-1,-2}:
            return True
        if s2.node in pickup_node:
            drop_off_node_need_to_visit = [s2.node+len(pickup_node)]
            drop_off_vec = s2.state_vec[0,4+len(pickup_node):]
            dense_array = drop_off_vec.toarray()[0]
            zero_indices = np.where(dense_array == 0)[0]
            for n in zero_indices:
                drop_off_node_need_to_visit.append(n+1+len(dropoff_node))
            print('drop_off_node_need_to_visit')
            print(drop_off_node_need_to_visit)
            if len(drop_off_node_need_to_visit)>self.MaxDepth:
                return False
            if len(drop_off_node_need_to_visit) >0:
                permutation_of_drop_off_node = list(permutations(drop_off_node_need_to_visit))
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
                        return True
                print('THIS SPOT LOOK failing in confimr due to here')
                input('not neccesaraylly wrong but check')
                return False
            else:
                return True
        elif s2.node in dropoff_node:
            
            drop_off_node_need_to_visit = []
            drop_off_vec = s2.state_vec[0,4+len(pickup_node):]
            dense_array = drop_off_vec.toarray()[0]
            zero_indices = np.where(dense_array == 0)[0]
            for n in zero_indices:
                if n+1+len(pickup_node) != s2.node:
                    drop_off_node_need_to_visit.append(n+1+len(dropoff_node))
            if len(drop_off_node_need_to_visit)>self.MaxDepth:
                return False
            if len(drop_off_node_need_to_visit) >0:
                permutation_of_drop_off_node = list(permutations(drop_off_node_need_to_visit))
            
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
                        return True
                print('123 failing in confimr due to here')

                return False
            else:
                a = self.actions[(s2.node,-2)][0]
                new_state = a.get_head_state(s2,s2.l_id)
                if new_state != None:
                    return True
                else:
                    print('1112 failing in confimr due to here')

                    return False
        else:
            input('state is not pickup or drop off node')
        #checkign my permuations
        #self.SLAI is teh shawn state generation structure
        #if its valdi then return true otehrwise return fasle
    
    def apply_node_min_term(self,in_state):
        print('hellow orld')
        s = in_state
        new_state_vec = self.elementwise_min_csr(s.state_vec, self.node_min_term_vec[s.node])
        if np.sum(np.abs(new_state_vec-s.state_vec)) > .00001:
            s2 = State(s.node, new_state_vec, s.l_id, s.is_source, s.is_sink)
            return s2
        else:
            return s   
    def jy_can_expand(self,s_origin,my_action):
        #return false if takign this action from theis state produces an ifeasible action
        #possibilites
        #possibility s_deestination is none
        s_des = my_action.get_head_state(s_origin,s_origin.l_id)
        if s_des == None:
            print('failing one ')
            s_origin.pretty_print_state()
            my_action.pretty_print_action()
            print('failing here 1')
            input('not wrong but look')
            return False
        this_state_vec = s_des.state_vec
        des_node = s_des.node
        try:
            if  des_node in self.dropoff_node and this_state_vec[0,self.resource_name_to_index[str(('may_avoid_dropoff',des_node))]] == 1:
                print('failing here 2')
                return False
        except:
            print('check here')
        #the my_action.node_head (meaning destination ) is a dropoff  and the MustAvoidDropOff is not active for it
        if self.confirm_if_state_possible(s_des) == False:
            print('failing here 3')
            return False
        #possibility confirm_if_state_possible returns false
        pick_up_vec = s_des.state_vec[0,4:4+len(self.pickup_node)] #NO CONSTANTS IN THE CODE PLEASE
        dense_array = pick_up_vec.toarray()[0]
        num_pickup = len(np.where(dense_array == 0)[0])
        if num_pickup > self.MaxDepth:# NO CONSTANTS IN THE CODE PLEASE
            print('failing here 4')

            return False
        #possibility:  there would be more than maxPickupsInstate pickups overall;
            #remember you have a reource for this. so this should be caught by get head state but do check
 
        return True
        #return true otherwise

    
    def update_action_subset_given_col(self):
        Q=self.SLAI
        if self.use_all_actions==False:
            self.Action_subset=Q.action_reasonable.copy()
            for i_ind in range(0,len(self.list_of_states_in_col_ordered)):
                for j_ind in range(i_ind+1,len(self.list_of_states_in_col_ordered)):
                    s1=    self.list_of_states_in_col_ordered[i_ind]
                    s2=    self.list_of_states_in_col_ordered[j_ind]

                    if (s1.node,s2.node) in Q.actions:#[s1.node,s2.node]:
                        my_actions_n1_n2=Q.actions[s1.node,s2.node]
                        self.Action_subset=self.Action_subset.union(my_actions_n1_n2)
        else:
            self.Action_subset=[]
            for (n1,n2) in self.actions:
                if n1>len(self.pickup_node)*2 or n2>len(self.pickup_node)*2:
                    continue
                
                for a in self.actions[n1,n2]:
                    self.Action_subset.append(a)
            print('len(Action_subset)')
            print(len(self.Action_subset))
            input('--')
    def updeate_action_from_nodes_subset(self):
        Q=self.SLAI
        self.actions_from_node_subset=dict()
        #self.actions_from_node_subset_pickup=dict()
        self.actions_from_node_subset_dest_dropoff=dict()
        #self.actions_from_node_subset_sink=dict()
        self.actions_from_node_subset_MINUS_dest_dropoff=dict()
        for n in Q.nodes:
            self.actions_from_node_subset[n]=[]
            self.actions_from_node_subset_MINUS_dest_dropoff[n]=[]
            self.actions_from_node_subset_dest_dropoff[n]=[]
        #print('actions_from_node_subset')
        #print(self.actions_from_node_subset)
        for a in self.Action_subset:
            my_origin=a.node_tail
            my_destination=a.node_head
            do_add=False
            if self.option_do_min_term:
                #shawn 
                if np.min(self.node_min_term_vec[a.node_tail].toarray()- a.min_resource_vec.toarray())>=-0.0001:
                    do_add=True
                    self.actions_from_node_subset[my_origin].append(a)
            else:
                do_add=True
            if do_add==True:
                #print('my_origin')
                #print(my_origin)
                #print('my_origin in Q.nodes')
                #print(my_origin in Q.nodes)
                self.actions_from_node_subset[my_origin].append(a)
                if my_destination in Q.dropoff_node:
                    self.actions_from_node_subset_dest_dropoff[my_origin].append(a)
                    #print('hihih')
                    #print('my_origin')
                    #print(my_origin)
                    #input('---')
                else:
                    self.actions_from_node_subset_MINUS_dest_dropoff[my_origin].append(a)
        
    def expand_state_given_action(self,s,my_act,orig_depth_s):

        can_expand=self.jy_can_expand(s,my_act)
        if can_expand==False:
            return False,None,None
        #\State  $s_2\leftarrow GetHeadState(s\rightarrow a)$
        my_head=my_act.get_head_state(s,s.l_id)
        my_new_depth=orig_depth_s-self.DepthUsed[my_act]
        #\If {$s_2== None$}
        did_make_term=False
        debug_on=True
        #\State Continue
        did_make_term=True
        #\EndIf
        if debug_on:
            if False==self.confirm_if_state_possible(my_head):
                my_head.pretty_print_state()
                input('error here 1')
        #\State $s_2.stateVec\leftarrow ElementwiseMin(s_2.stateVec,s2.node.NodeMinTermVec)$
        if self.option_do_min_term:
            my_head=self.apply_node_min_term(my_head)
        if debug_on:
            if False==self.confirm_if_state_possible(my_head):
                my_head.pretty_print_state()

                input('error here 2')
        if my_head==None and did_make_term==True:
            input('error here 332')
        if my_act.node_head==-2:

            print('did_make_term')
            print(did_make_term)
            s.pretty_print_state()
            input('-  at termination -2 node --')
        return did_make_term,my_head,my_new_depth

    def get_must_drop_off_including_current(self,s):
        Q=self.SLAI
        drop_off_node_need_to_visit=[]
        if s.node in self.pickup_node:
            print('part one ')
            drop_off_node_need_to_visit = []
            drop_off_vec = s.state_vec[0,4+len(self.pickup_node):]
            dense_array = drop_off_vec.toarray()[0]
            zero_indices = np.where(dense_array == 0)[0]
            for n in zero_indices:
                drop_off_node_need_to_visit.append(n+1+len(self.dropoff_node))
            drop_off_node_need_to_visit.append(s.node+len(self.dropoff_node))
        elif s.node in self.dropoff_node:
            print('part two ')

            drop_off_node_need_to_visit = []
            drop_off_vec = s.state_vec[0,4+len(self.pickup_node):]
            dense_array = drop_off_vec.toarray()[0]
            zero_indices = np.where(dense_array == 0)[0]
            for n in zero_indices:
                if n+1+len(self.pickup_node) != s.node:
                    drop_off_node_need_to_visit.append(n+1+len(self.dropoff_node))
        
        # may_avoid_dropoff=s.state_vec[4+self.num_pickups:]
        # must_dropoff=np.nonzero(may_avoid_dropoff<0.5)
        # must_dropoff=may_avoid_dropoff_list+self.num_pickups
        # if s.node in Q.pickup_node:
        #     must_dropoff.append(s.node+self.num_pickups)
        return drop_off_node_need_to_visit
    def get_actions_from_node_subset(self,s):
        Q=self.SLAI
        actions_use=self.actions_from_node_subset_MINUS_dest_dropoff[s.node].copy()
        must_dropoff=self.get_must_drop_off_including_current(s)
        #print('s is')
        #s.pretty_print_state()
        #print('must_dropoff')
        #print(must_dropoff)
        #input('---')
        for n in must_dropoff:
            my_act_list=Q.actions[(s.node,n)]
            for my_act in my_act_list:
                actions_use.append(my_act)
        return actions_use
    def gen_all_states_naive(self):

        Q=self.SLAI
        
        while len(self.my_sorted)>0:
            #State $s\leftarrow \mbox{arg} \max_{s\in StatesCanExpand}State2Depth(s)$
            #Pop(s)$ from $StatesCanExpand$.  Always select to expand the term with $State2Depth$ 

            s=self.my_sorted.pop_max()
            orig_depth_s=self.State2Depth[s]

            actions_use=self.get_actions_from_node_subset(s)
            print('working on state ')
            s.pretty_print_state()
            must_dropoff=self.get_must_drop_off_including_current(s)
            print('must_dropoff')
            print(must_dropoff)
            input('---')
            #print('actions above')
            for my_act in actions_use:
                my_act.pretty_print_action()
            input('showing actions')
            for my_act in actions_use:
                #print('working on state ')
                #s.pretty_print_state()
                #my_act.pretty_print_action()

                [did_make_new_state,my_head,my_new_depth]=self.expand_state_given_action(s,my_act,orig_depth_s)
                #print('did_make_new_state')
                #print(did_make_new_state)
                #print('my_new_depth')
                #print(my_new_depth)
                #if did_make_new_state==True:
                #    my_head.pretty_print_state()
                #input('---')

                if did_make_new_state==True  and my_head not in self.State2Depth and my_new_depth>-0.5:
                    print('doing addition ')
                    self.State2Depth[my_head]=my_new_depth
                    #print('my_new_depth')
                    #print(my_new_depth)
                    #print('type(my_new_depth)')
                    #print(type(my_new_depth))
                    #print('type(my_head)')
                    #print(type(my_head))
                    self.my_sorted.insert(my_head,my_new_depth)
                    if my_head==None:
                        input('error here')
                    self.all_states.append(my_head)
                else:
                    print('not doing addition')
                print('len(self.all_states)')
                print(len(self.all_states))
                print('len(self.my_sorted)')
                print(len(self.my_sorted))
                #input('done iter')
        #print('at end of gen naive self.all_states')
        #print(self.all_states)
        #input('---')
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
