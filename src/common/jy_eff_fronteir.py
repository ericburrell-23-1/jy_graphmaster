import bisect
import numpy as np

from numpy import zeros, ones
from src.common.helper import Helper
from scipy.sparse import csr_matrix
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from src.common.state import State
from src.common.action import Action
import numpy as np

class jy_efficient_frontier:
    def __init__(self,all_nodes):
        self.all_nodes=all_nodes
        self.node_2_eff_fronteir=dict()
        for my_node in all_nodes:
            self.node_2_eff_fronteir[my_node]=set([])

    def is_in_fronteir(self,input_label):
        if input_label in self.node_2_eff_fronteir[input_label.node]:
            return True
        return False

    def get_lowest_lb(self):
        lowest_lb=np.inf
        for my_node in self.all_nodes:
            for input_label in self.node_2_eff_fronteir[my_node]:
                if lowest_lb>input_label.lb:
                    lowest_lb=input_label.lb
                #lowest_lb=np.min(lowest_lb,)
                #self.node_2_eff_fronteir[my_node]
        return lowest_lb
    def alter_fronteir_given_new_element(self,new_label):

        is_in_frontier=True
        for old_label in self.node_2_eff_fronteir[new_label.node]:
            does_dom,does_equal = old_label.this_label_dominates_input(new_label)
            if does_dom==True or does_equal==True :
                is_in_frontier=False
                #print('old_label')
                #print(old_label.all_nodes_ordered)
                #print('new_label')
                #print(new_label.all_nodes_ordered)
                #input('--not adding --')
                break
        #print('is_in_frontier')
       # print(is_in_frontier)

        if is_in_frontier==True:
            labels_input_dominates=[]

            for old_label in self.node_2_eff_fronteir[new_label.node]:
                does_dom,does_equal=new_label.this_label_dominates_input(old_label)
                labels_input_dominates.append(old_label)
            
            for old_label in labels_input_dominates:#self.node_2_eff_fronteir[new_label.node]:
                self.node_2_eff_fronteir[new_label.node].remove(old_label)
            
            self.node_2_eff_fronteir[new_label.node].add(new_label)
