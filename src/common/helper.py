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