from typing import List, Any

def get_list_item_safe(ls: List[Any], index: int) -> Any:
        """
        Function to get an item from a list at a specified index. 
        If the index is out of bounds of the list, the first or last item
        of the list respectively is returned.

        Args:
            ls (List[Any]): List to get item from
            index (int): Index of item to get
        
        Returns:
            Any: Item at specified index or first/last item if index is out of bounds
        """
        if index <= 0:
            return ls[0]
        if index >= len(ls):
            return ls[-1]
        
        return ls[index]