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



class jy_sortedObject_list:
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

