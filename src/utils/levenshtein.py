class Levensthein():
    """
    A class that offers several functions related to the Levensthein distance between to strings.

    The Levenshtein distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to change one string into
    the other.

    Usage:
        ld = Levenshtein()
        distance = ld.levensthein_distance('ANANAS', 'BANANE')
        print(distance)  # Output: 3

        mods = ld.levensthein_mods('ANANAS', 'BANANE')
        print(mods)  # Output: ['Delete B', 'Insert A', 'Substitute E with S'] 

    Attributes:
        None

    Methods:
        levensthein_distance: Calculates the Levenshtein distance between two strings.
        levensthein_mods: Calculates the modifications on the strings to achieve minimal distance
    """
    def __init__(self):
        pass

    def __levensthein_preprocess_sentence(self, sentence):
        """
        Private function to preprocess the input (sentence as a string) 
        by splitting the sentence into a list of words.

        Args:
            sentence (str): The input string to be preprocessed
        
        Returns: 
            Preprocessed string
        """

        # split sentence into list of words
        sentence_list = sentence.split()

        return sentence_list

    def levensthein_distance(self, a: str, b: str) -> int:
        """
        Calculates the Levenshtein distance between two strings on the word level.

        Args:
            a (str): The first sentence.
            b (str): The second sentence.

        Returns:
            int: The Levenshtein distance between the two strings.
        """
        # preprocess words
        a_list = self.__levensthein_preprocess_sentence(a)
        b_list = self.__levensthein_preprocess_sentence(b)

        J = len(a_list)
        K = len(b_list)

        l_matrix = self.__levensthein_matrix(a_list, b_list)
        return l_matrix[J][K]

    def __levensthein_matrix(self, a, b):
        """
        Private function to calculate the dynamic-programming table for levensthein distance (word level)

        Args:
            a (List[str]): The first sentence encoded by a list, where each list entry represents one word.
            b (List[str]): The second sentence encoded by a list, where each list entry represents one word.

        Returns:
            List[int][int]: The "levensthein-matrix".
        """
        J = len(a)
        K = len(b)

        # first coordinate: position in a
        # second coordinate: position in b
        # initializing matrix with 0's
        l_matrix = [(K +1) * [0] for _ in range(J + 1)]
        
        # fill in base cases (outer rows/columns)
        for j in range(J+1):
            l_matrix[j][0] = j 
        for k in range(K+1):
            l_matrix[0][k] = k
        
        for j in range(1, J+1):
            for k in range(1, K+1):
                # Case: Matching
                if a[j-1] == b[k-1]:
                    l_matrix[j][k] = l_matrix[j-1][k-1]
                    continue

                # Case: No matching, check for subst/ins/del
                substitute = l_matrix[j-1][k-1] + 1
                insertion = l_matrix[j][k-1] + 1
                deletion = l_matrix[j-1][k] + 1

                l_matrix[j][k] = min(substitute, insertion, deletion)
        
        return l_matrix

    def levensthein_mods(self, a, b):
        """
        Calculates the modifications that have to be done to get from first string to second string on word level.

        Args:
            a (str): The first sentence.
            b (str): The second sentence.

        Returns:
            List[str]: List of the modifications
        """
        a = self.__levensthein_preprocess_sentence(a)
        b = self.__levensthein_preprocess_sentence(b)

        J = len(a)
        K = len(b)
        # first coordinate: position in a
        # second coordinate: position in b
        l_matrix = self.__levensthein_matrix(a, b)
        mods = []
        j = J
        k = K
        # follow the path from end-corner to start-corner by always going  to neighbor with minimal value
        while j != 0 or k != 0:
            # if start of first word is reached, the remaining letters from second word need to be inserted
            if j == 0:
                mods = [f"Insert '{b[k-1]}'"] + mods
                k -= 1
                continue
            # if start of second word is reached, the remaining letters from first word need to be deleted
            if k == 0:
                mods = [f"Delete '{a[j-1]}'"] + mods
                j -= 1
                continue
            
            # Case: Matching; no modification needed
            if a[j-1] == b[k-1]:
                k -= 1
                j -= 1
                continue

            # Case: No matching; Check which modification is needed
            substitute = l_matrix[j-1][k-1] + 1
            insertion = l_matrix[j][k-1] + 1
            deletion = l_matrix[j-1][k] + 1

            min_value = min(substitute, insertion, deletion)

            # Check which modification has minimal value, add corresponding modification and change j,k
            if min_value == substitute:
                mods = [f"Substitute '{a[j-1]}' with '{b[k-1]}'"] + mods
                j -= 1
                k -= 1
            elif min_value == insertion:
                mods = [f"Insert '{b[k-1]}'"] + mods
                k -= 1
            elif min_value == deletion:
                mods = [f"Delete '{a[j-1]}'"] + mods
                j -= 1
            
        return mods
    
if __name__ == "__main__":
    lv = Levensthein()
    print("Testing simple cases...")
    assert lv.levensthein_distance("The dog is under the table", "The dog is the fable") == 2
    assert lv.levensthein_distance("hello world", "hello duck") == 1
    assert lv.levensthein_distance("i like monthy python", "i like python")

    print("Testing edge cases...")
    assert lv.levensthein_distance("", "The dog is the fable") == 5
    assert lv.levensthein_distance("The dog is under the table", "") == 6
    assert lv.levensthein_distance("", "") == 0

    print("Tests succesful!")