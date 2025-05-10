import bisect

class jy_sortedObject_list:
    """Maintains a sorted list of objects based on an associated lexicographic tuple of keys."""
    
    def __init__(self):
        self.keys = []      # List of tuple keys (used for lexicographic sorting)
        self.objects = []   # List of associated objects

    def insert(self, obj, keys):
        """
        Inserts an object while keeping the list sorted by lexicographic order of keys.
        
        Args:
            obj: The object to insert.
            keys: A tuple of keys used for sorting.
        """
        #print('self.keys')
        #print(self.keys)
        #print('keys')
        #print(keys)
        #input('---')
        if obj not in self.objects:
            index = bisect.bisect_left(self.keys, keys)
            self.keys.insert(index, keys)
            self.objects.insert(index, obj)

    def pop(self):
        """
        Removes and returns the object with the smallest keys (lexicographically).
        
        Returns:
            The object with the smallest keys.
            
        Raises:
            IndexError: If the list is empty.
        """
        if not self.objects:
            raise IndexError("Pop from empty SortedObjectList")
        self.keys.pop(0)
        return self.objects.pop(0)

    def pop_max(self):
        """
        Removes and returns the object with the largest keys (lexicographically).
        
        Returns:
            The object with the largest keys.
            
        Raises:
            IndexError: If the list is empty.
        """
        if not self.objects:
            raise IndexError("Pop from empty SortedObjectList")
        self.keys.pop()
        return self.objects.pop()

    def __len__(self):
        return len(self.objects)

    def __repr__(self):
        return str(list(zip(self.keys, self.objects)))
