import operator
import numpy as np
from scipy.sparse import csr_matrix
class Helper:
    @staticmethod
    def subset_where_z_in_Y(z, X, Y):
        """
        Computes the subset of X such that z is in Y[x] for each x in X.

        Parameters:
        z (any): The element to check for in the sets.
        X (set): The set of keys to consider.
        Y (dict): A dictionary where values are sets.

        Returns:
        set: The subset of X where z is present in Y[x].
        """
        return {x for x in X if x in Y and z in Y[x]}

    @staticmethod
    def union_of_sets(X, Y):
        """
        Computes the union of sets X[y] for all y in Y.

        Parameters:
        X (dict): Dictionary where values are sets.
        Y (set): A set of keys to look up in X.

        Returns:
        set: The union of all sets X[y] for y in Y.
        """
        return set().union(*(X[y] for y in Y if y in X))
    
    @staticmethod
    def operate_on_chainmaps(cm1, cm2, op=operator.sub):
        """
        Operate on values from two ChainMaps/dictionaries with the same keys.

        :param cm1: The first ChainMap (or dict)
        :param cm2: The second ChainMap (or dict)
        :param op:  A binary operator function (default is operator.sub for subtraction).
                    Could be operator.add, operator.mul, or a custom lambda, e.g. lambda x,y: x+y
        :return:    A dict with {key: op(value_in_cm1, value_in_cm2)} for each key
        """
        result = {}
        for key in cm1:
            # Since we assume cm1 and cm2 have the same keys, just apply the operator
            result[key] = op(cm1[key], cm2[key])
        return result
    
    @staticmethod
    def dict_2_vec(key_2_index,vec_sz,key_to_value):
 
        indices = np.array([key_2_index[k] for k in key_to_value], dtype=int)
        data = np.array([key_to_value[k] for k in key_to_value])
        # Create a 1 x num_resources sparse row vector.
        my_vec = csr_matrix((data, (np.zeros(len(data), dtype=int), indices)),shape=(1,vec_sz))                
 
 
        return indices,my_vec