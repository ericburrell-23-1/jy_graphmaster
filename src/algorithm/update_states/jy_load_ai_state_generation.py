

#jy_load_ai_state_generation

class jy_make_load_ai_states():

    def __init__(self,shawn_LoadAI_state_input,list_of_action_in_col_ordered,list_of_states_in_col_ordered):
        print('hello world')
        self.SLAI=shawn_LoadAI_state_input
        self.list_of_action_in_col_ordered=list_of_action_in_col_ordered
        self.list_of_states_in_col_ordered=list_of_states_in_col_ordered
        self.node_2_actions=node_2_actions
        self.gen_all_states_naive()
    def return_solution(self):
        return self.all_states


    def confirm_if_state_possible(self,candid_state):
        print('hello world')
        #checkign my permuations
        #self.SLAI is teh shawn state generation structure
        #if its valdi then return true otehrwise return fasle
    
    def apply_node_min_term(self,in_state):
        print('hellow orld')
        #return the new state if it need
        return projected_state
    def gen_all_states_naive(self):

        Q=self.SLAI
        option_do_min_term=True
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
                Action_subset=Q.action_reasonable.copy()
                for i_ind in range(0,len(self.list_of_states_in_col_ordered)):
                    for j_ind in range(i_ind+1,len(self.list_of_states_in_col_ordered)):
                        s1=    self.list_of_states_in_col_ordered[i_ind]
                        s2=    self.list_of_states_in_col_ordered[j_ind]
                        my_actions_n1_n2=Q.actions[s1.node,s2.node]
                        Action_subset=Action_subset.union(my_actions_n1_n2)
                self.states_can_expand=[]
                if option_do_min_term==True:
                    tmp=[]
                    for s in self.list_of_states_in_col_ordered:
                        s2=self.apply_node_min_term(s)
                        tmp.append(s2)
                        self.states_can_expand.append(s2)
                    self.list_of_states_in_col_ordered=tmp
                #State $State2Depth(s)\leftarrow MaxDepth$ for all $s\in StatesCanExpand$
                self.State2Depth=dict()
                for s in self.states_can_expand:
                    
                \State $ActionsFromNode(n)\leftarrow \{ \}$ for all $n\in Nodes$
                \For{$a \in ActionSubset$}
                \If{$a.origin.NodeMinTermVec\geq a.minTermInput$}
                \State $ActionsFromNode(a.origin)\leftarrow a$
                \EndIf
                \EndFor
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