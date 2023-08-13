import pickle
import os
import gzip
from typing import List

START = 0  # 0 -> start of sentence <s>
UNK = 2   # 2  -> unknown word <UNK>
END = 1    # 1  -> end of sentende </s>

class Dictionary(): 
    """
    Class for creating and loading vocabulary dictionaries to bijectively map string tokens to integer indices.
    The dictionary is stored as a pickle file in the data/dictionaries directory.
    """

    __mapping = dict()
    __path_to_file = ""
  
    def __init__(self, name: str = "", save_model: bool = True) -> None:
        """
        Initializes the dictionary with the specified name. If a dictionary with the 
        specified name exists, it will be loaded, else a new dictionary will be created.

        Args:
            name (str): The name of the dictionary to load or create 
        """
        
        # Get the path to the dictionary file
        """
        Old version:

        dir_path = os.path.dirname(os.path.realpath(__file__))
        path_to_dir = os.path.abspath(os.path.join(os.path.abspath(os.path.join(dir_path,os.pardir)),os.pardir))
        self.__path_to_file = os.path.join(path_to_dir, "data/dictionaries/dict_" + name + ".pkl")
        """
        self.__path_to_file = "../data/dictionaries/dict_" + name + ".pkl"

        # Check if file exists
        if os.path.exists(self.__path_to_file) and save_model:
            # Load the dictionary
            try:
                with open(self.__path_to_file, "rb") as savefile:
                    self.__mapping = pickle.load(savefile)
                print("Dictionary has been read.")
            except (IOError, OSError) as e:
                print(f"Error reading dictionary: {str(e)}")
                self.__mapping = {"<UNK>": UNK, "<s>": START, "</s>": END}
        else:
            # Create a new dictionary
            self.__mapping = {"<UNK>": UNK, "<s>": START, "</s>": END}

        # Print the dictionary
        print(self.__mapping)
    
    def getIndex(self, token: str) -> int:
        """
        Returns the index of the token if it exists, else returns UNK index

        Args:
            key (String): The token to look for

        Returns:
            int: The int-index for the token
        """
        if(self.__mapping.get(token) is None): 
            return UNK
        return self.__mapping[token]
    
    def getToken(self, index: int) -> str:
        """
        Returns the token of the index if it exists, else returns <UNK>
        """
        inverse_map = dict((v,k) for k,v in self.__mapping.items())
        if index in inverse_map.keys():
            return inverse_map[index]
        return "<UNK>"
    
    def update(self, lines: List[List[str]]) -> None:
        """
        Updates the dictionary with the specified list of sentences

        Args:
            lines (List[List[str]]): List of sentences to update the dictionary with, where each sentence
                is encoded as a list of tokens        
        """

        # Catch empty list
        if not lines:
            return
        
        # Get the current keys in the dictionary
        curr_tokens = self.__mapping.keys()
        
        # Split the sentences into words
        word_list = [word for sentence in lines for word in sentence]

        """
        Old version:
        
        split_lines = [line.split(" ") for line in lines]
        word_list = [word for sentence_list in split_lines for word in sentence_list]
        """

        # Determine actually new elements
        intersect = set(curr_tokens).intersection(word_list)
        new_elements = {word for word in word_list if word not in intersect}

        # Add new elements to the dictionary
        offset = len(curr_tokens)
        for index, word in enumerate(new_elements):
            self.__mapping[word] = index + offset
             
    def apply_mapping(self, param: List[List[str]]) -> List[List[int]]:
        """
        Applies the mapping to the specified list of sentences and returns the result.

        Args:
            param (List[List[str]]): List of sentences to apply the mapping to, where each
                sentence is encoded as a list of tokens

        Returns:
            List[List[int]]: List of int-sentences, where each integer represents
                the index of the token in the input sentence
        
        """
        if not param:
            return []
        
        return [[self.getIndex(token) for token in sentence] for sentence in param]

        # Old version
        source = param
        for i in range(len(source)):
            for j in range(len(source[i])): 
                source[i][j] = self.getIndex(source[i][j])
        return source
    
    def getKeys(self):
        """
        Returns the list of all tokens in the vocabulary
        
        Returns:
            List[str]: List of all tokens in the vocabulary
        """
        return (self.__mapping.keys())
    
    def reset(self): 
        """
        Resets the dictionary to its initial state (only contains <UNK>, <s> and </s>) and saves it
        """
        self.__mapping = dict({"<UNK>": UNK ,"<s>": START, "</s>" : END})
        self.save()
    
    def save(self): 
        """
        Saves the dictionary to a pickle file
        """
        with open(self.__path_to_file, "wb+") as savefile:
            pickle.dump(self.__mapping, savefile, protocol=pickle.HIGHEST_PROTOCOL)
    
    def get_vocab_size(self):
        """
        Returns the size of the vocabulary

        Returns:
            int: The size of the vocabulary
        """
        return len(self.__mapping)
            
#x = Dictonary("EN")
#print(x.getKey(10826))
#with gzip.open(r"C:\Users\GlenGrant\Desktop\RWTH\4. Semester\Maschinelle Uebersetzung (Praktikum)\Projekt\machine-translation\data\data_v2\multi30k.en.gz", "rt", encoding="utf-8") as source:
#    content=source.readlines()
#    x.update(content)
#    print(x.getKeys())
#    x.save()
#x = Dictonary("EN")
#print(x.apply_mapping(["hello this is a test"]))

        
    