
from itertools import permutations
import numpy as np
from scipy.sparse import csr_matrix
from src.common.state import State
#jy_load_ai_state_generation

class jy_make_load_ai_states():

    def __init__(self,shawn_LoadAI_state_input,list_of_action_in_col_ordered,list_of_states_in_col_ordered):
        print('hello world')
        self.SLAI=shawn_LoadAI_state_input
        self.list_of_action_in_col_ordered=list_of_action_in_col_ordered
        self.list_of_states_in_col_ordered=list_of_states_in_col_ordered
        self.node_2_actions=node_2_actions
        self.shawn_LoadAI_state_input = shawn_LoadAI_state_input
        self.node_min_term_vec = shawn_LoadAI_state_input['node_min_term_vec']
        self.gen_all_states_naive()
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
    def gen_all_states_naive(self):

        Q=self.SLAI
        option_do_min_term=True
        my_sorted=SortedObjectList_2()
        #        \State $MaxDepth\leftarrow $ User
        MaxDepth=self.jy_options['max_pickups_in_a_route']#S.max_depth
        #        \State $DepthUsed(a)\leftarrow $User Defined 
        DepthUsed=dict()
        for a in Q.actions:
            node_origin=a.node_tail
            node_destination=a.node_head
            if node_destination in Q.pickup_nodes:
                DepthUsed[a]=1
            else:
                DepthUsed[a]=0
        


            #State $myInitStates \leftarrow $ From User; By Calling Pricing.  These are the states in that column.  Ordered in terms of execution
            #JY DONE
                #%\State $\beta \leftarrow $User 
            # JY GIVE SHAWN CODE beta=Q.generate_beta
                #\State $n.NodeMinTermVec\leftarrow$ User (get from $\beta$ which we will compjute 
                #separately)
            #JY_GIVE SHAWN CODE
                #\State $ActionsReasonalbe\leftarrow $ User; Can be all actions or just actions  or just actions that we know are not dumb.
                #\State $ActionsSubset\leftarrow ActionsReasonalbe.copy()$
                #\For{$i=0:Len(myInitStates)$, $j=i+1:len(myInitStates)$}
                #    \State $s_1\leftarrow myInitStates[i]$ 
                #    \State $s_2\leftarrow myInitStates[j]$ 
                #    \State $ActionsSubset\leftarrow ActionsSubset+AllActions(node(s_1)\rightarrow node(s_2))$
                #\EndFor
                self.Action_subset=Q.action_reasonable.copy()
                for i_ind in range(0,len(self.list_of_states_in_col_ordered)):
                    for j_ind in range(i_ind+1,len(self.list_of_states_in_col_ordered)):
                        s1=    self.list_of_states_in_col_ordered[i_ind]
                        s2=    self.list_of_states_in_col_ordered[j_ind]
                        my_actions_n1_n2=Q.actions[s1.node,s2.node]
                        self.Action_subset=self.Action_subset.union(my_actions_n1_n2)
                self.states_can_expand=[]
                if option_do_min_term==True:
                    tmp=[]
                    for s in self.list_of_states_in_col_ordered:
                        s2=self.apply_node_min_term(s)
                        tmp.append(s2)
                        self.states_can_expand.append(s2)
                        my_sorted.insert(s2,MaxDepth)
                        
                    self.list_of_states_in_col_ordered=tmp
                #State $State2Depth(s)\leftarrow MaxDepth$ for all $s\in StatesCanExpand$
                self.State2Depth=dict()
                for s in self.states_can_expand:
                    self.State2Depth[s]=MaxDepth
                
                #\State $ActionsFromNode(n)\leftarrow \{ \}$ for all $n\in Nodes$
                #\For{$a \in ActionSubset$}
                #\If{$a.origin.NodeMinTermVec\geq a.minTermInput$}
                #\State $ActionsFromNode(a.origin)\leftarrow a$
                #\EndIf
                #\EndFor

                self.actions_from_node=dict()
                for n in Q.nodes:
                    self.actions_from_node[n]=[]
                for a in self.Action_subset:
                    my_origin=a.node_tail
                    if option_do_min_term:
                        if np.min(a.origin.NodeMinTermVec- a.minTermInput>=-0.0001):
                            self.actions_from_node[my_origin].append(a)
                    else:
                        self.actions_from_node[my_origin].append(a)
                while len(StatesCanExpand>0):

                \While{$|StatesCanExpand|>0$}
                    \State $s\leftarrow \mbox{arg} \max_{s\in StatesCanExpand}State2Depth(s)$
                    \State $StatesCanExpand\leftarrow StatesCanExpand-s$
                    %Pop(s)$ from $StatesCanExpand$.  Always select to expand the term with $State2Depth$ 
                    \For{$a\in ActionsFromNode(s.node)$}
                    \State  $s_2\leftarrow GetHeadState(s\rightarrow a)$
                        \If {$s_2== None$}
                        \State Continue
                        \EndIf
                        \If{$UserIgnoreStateAction(s_2,a)=True$.  User defined function; always False by default}
                        \State Continue
                        \EndIf
                        \State $s_2.stateVec\leftarrow ElementwiseMin(s_2.stateVec,s2.node.NodeMinTermVec)$
                        \If{$s_2 \notin State2Depth$ and $State2Depth(s)>0$}
                        \State $State2Depth(s_2)\leftarrow State2Depth(s)-DepthUsed(a)$
                        \State $StatesCanExpand\leftarrow StatesCanExpand+s_2$
                        \EndIf
                    \EndFor
                \EndWhile
            \end{algorithmic}
        \end{algorithm}
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


class SortedObjectList_2:
    """Maintains a sorted list of objects based on an associated scalar value."""
    
    def __init__(self):
        self.values = []  # List of scalar values (used for sorting)
        self.objects = []  # List of associated objects

    def insert(self, obj, value):
        """Inserts an object while keeping the list sorted by value."""
        index = bisect.bisect_left(self.values, value)  # Find insertion index
        self.values.insert(index, value)  # Insert value in sorted order
        self.objects.insert(index, obj)  # Insert object in corresponding position

    def pop(self):
        """Removes and returns the object with the smallest value."""
        if not self.objects:
            raise IndexError("Pop from empty SortedObjectList")
        self.values.pop(0)  # Remove first (smallest) value
        return self.objects.pop(0)  # Remove and return first object

    def pop_max(self):
        """Removes and returns the object with the largest value."""
        if not self.objects:
            raise IndexError("Pop from empty SortedObjectList")
        self.values.pop()  # Remove last (largest) value
        return self.objects.pop()  # Remove and return last object

    def __len__(self):
        return len(self.objects)

    def __repr__(self):
        return str(list(zip(self.values, self.objects)))  # Show sorted pairs
