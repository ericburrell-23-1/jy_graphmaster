import numpy as np
from scipy.sparse import csr_matrix

# Import the State class (assuming it's saved in state.py)
# In a real scenario, you would import from your module
# from state import State

# For testing, we'll just paste the State class here
class State:
    def __init__(self, node: int, state_vec: csr_matrix, l_id, is_source, is_sink):
        self.node = node
        # Ensure state_vec is a CSR matrix with integer data type
        self.state_vec = state_vec.tocsr().astype(int)
        self.l_id = l_id
        self.is_source = is_source
        self.is_sink = is_sink
        
        # Calculate state_id after initialization
        self._compute_state_id()
    
    def _compute_state_id(self):
        """Compute a deterministic hash for this state."""
        # Create a tuple of the non-matrix components
        base_tuple = (self.node, self.is_sink, self.is_source, self.l_id)
        
        # Get a deterministic hash of the CSR matrix
        matrix_hash = self._csr_matrix_hash()
        
        # Combine everything into a single hash
        combined = hash(base_tuple) ^ matrix_hash
        self.state_id = combined
    
    def _csr_matrix_hash(self):
        """
        Compute a deterministic hash for the CSR matrix.
        This normalizes the representation first to ensure that
        equivalent matrices hash to the same value.
        """
        # Convert to canonical form (sorts indices)
        matrix = self.state_vec.copy()
        matrix.sum_duplicates()
        matrix.sort_indices()
        
        # Extract matrix components
        data = tuple(matrix.data)
        indices = tuple(matrix.indices)
        indptr = tuple(matrix.indptr)
        shape = matrix.shape
        
        # Create a hashable representation
        matrix_tuple = (shape, data, indices, indptr)
        return hash(matrix_tuple)
    
    def __eq__(self, other):
        """Check if two State objects are equal."""
        if other is None or not isinstance(other, State):
            return False
        
        # Check non-matrix attributes first (faster)
        if (self.node != other.node or 
            self.is_sink != other.is_sink or 
            self.is_source != other.is_source or 
            self.l_id != other.l_id):
            return False
            
        # Check if matrices are equal using our helper method
        return self._csr_matrices_equal(self.state_vec, other.state_vec)
    
    def __hash__(self):
        """Return the precomputed hash value."""
        return self.state_id
    
    def _csr_matrices_equal(self, A, B):
        """
        Check if two CSR matrices represent the same mathematical matrix.
        This handles different internal representations of the same matrix.
        """
        # Check shapes
        if A.shape != B.shape:
            return False
        
        # Convert both to canonical form
        A_copy = A.copy()
        B_copy = B.copy()
        A_copy.sum_duplicates()
        B_copy.sum_duplicates()
        A_copy.sort_indices()
        B_copy.sort_indices()
        
        # Check components
        if not np.array_equal(A_copy.indptr, B_copy.indptr):
            return False
        if not np.array_equal(A_copy.indices, B_copy.indices):
            return False
        if not np.array_equal(A_copy.data, B_copy.data):
            return False
            
        return True
    
    def pretty_print_state(self):
        print('state description')
        print('l_id:  ' + str(self.l_id))
        print('node:  ' + str(self.node))
        print('stateVec:   ' + str(self.state_vec.toarray()))
        print('state_id:  ' + str(self.state_id))

# Create test case for the scenario you described
# Create two identical sparse matrices
data1 = np.array([1, 1])
indices1 = np.array([0, 1])
indptr1 = np.array([0, 2])
shape1 = (1, 4)

data2 = np.array([1, 1])  
indices2 = np.array([0, 1])
indptr2 = np.array([0, 2])
shape2 = (1, 4)

# Create CSR matrices
matrix1 = csr_matrix((data1, indices1, indptr1), shape=shape1)
matrix2 = csr_matrix((data2, indices2, indptr2), shape=shape2)

# Create two State objects with identical attributes
state1 = State(node=1, state_vec=matrix1, l_id=1, is_source=False, is_sink=False)
state2 = State(node=1, state_vec=matrix2, l_id=1, is_source=False, is_sink=False)

# Print the states
print("State 1:")
state1.pretty_print_state()
print("\nState 2:")
state2.pretty_print_state()

# Check equality
print("\nTesting equality:")
print(f"state1 == state2: {state1 == state2}")
print(f"hash(state1) == hash(state2): {hash(state1) == hash(state2)}")

# Test set behavior
state_set = set()
state_set.add(state1)
state_set.add(state2)
print(f"\nNumber of states in set: {len(state_set)}")

# Alternative representation for the same matrix (different internal structure but same mathematical matrix)
data3 = np.array([1, 1])
indices3 = np.array([1, 0])  # Indices in different order
indptr3 = np.array([0, 2])
shape3 = (1, 4)
matrix3 = csr_matrix((data3, indices3, indptr3), shape=shape3)

# Create another State with the same values but different matrix representation
state3 = State(node=1, state_vec=matrix3, l_id=1, is_source=False, is_sink=False)

print("\nState 3 (different representation of same matrix):")
state3.pretty_print_state()

# Test equality with state3
print("\nTesting equality with alternative representation:")
print(f"state1 == state3: {state1 == state3}")
print(f"hash(state1) == hash(state3): {hash(state1) == hash(state3)}")

# Add to set and check size
state_set.add(state3)
print(f"Number of states in set after adding state3: {len(state_set)}")