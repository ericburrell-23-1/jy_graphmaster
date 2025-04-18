import operator
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
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
    def dict_2_vec(key_2_index, vec_sz, key_to_value):
        # Filter keys to include only those with non-zero values
        filtered_keys = [k for k in key_to_value if key_to_value[k] != 0]
        # Build indices and data arrays from the filtered keys
        indices = np.array([key_2_index[k] for k in filtered_keys], dtype=int)
        data = np.array([key_to_value[k] for k in filtered_keys])
        # Create a 1 x vec_sz sparse row vector
        my_vec = csr_matrix((data, (np.zeros(len(data), dtype=int), indices)), shape=(1, vec_sz))
        return indices, my_vec
    def partial_map_2_indices_applied(key_2_index,key_to_value):
        #take in thej partial map and produce the terms where the min operator is applied
        indices = np.array([key_2_index[k] for k in key_to_value], dtype=int)
        #data = np.array([key_to_value[k] for k in key_to_value])
        # Create a 1 x num_resources sparse row vector.
        #my_vec = csr_matrix((data, (np.zeros(len(data), dtype=int), indices)),shape=(1,vec_sz))                

        return indices
    
    def LOAD_AI_partial_map_2_indices_applied(key_2_index,destination_node,origin_node):
    
        key_list=["time","volume","weight","max_combined_loads"]
        if origin_node in self.pickup_nodes:
            key_list.append("may_pickup"+origin_node)
            key_list.append("may_avoid_dropoff"+origin_node)
        indices = np.array([key_2_index[k] for k in key_to_value], dtype=int)
        return indices

        #indices_apply_min_to=Helper.LOAD_AI_partial_map_2_indices_applied(self.resource_name_to_index,destination_node,origin_node)


    def merge_two_dict(dict1, dict2):
        """
        Merges two time profile dictionaries, adding values for keys that appear in both.
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary
            
        Returns:
            dict: Combined dictionary with summed values for shared keys
        """
        merged = defaultdict(float)
        
        # Add all entries from the first dictionary
        for key, value in dict1.items():
            merged[key] += value
        
        # Add all entries from the second dictionary, summing values for shared keys
        for key, value in dict2.items():
            merged[key] += value
        
        return merged