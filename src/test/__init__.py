import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print(sys.path)
from src.common.state import State
import numpy as np
from scipy.sparse import csr_matrix


arr = np.array([1,2,2,1])
csr = csr = csr_matrix(arr)
state1 = State(1,csr,1,None,None)
state2 = State(1,csr,1,None,None)
print(state1.state_id)
print(state2.state_id)
state_set = set()
state_set.add(state1)
state_set.add(state2)
print(state_set)