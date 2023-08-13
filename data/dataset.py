import sys
sys.path.append('../..')

from src.preprocessing.batching import Batcher
from src.preprocessing.dictionary import Dictionary
from src.utils.byte_pair_encoding import BytePairEncoder
from src.pickle_loader import PickleLoader
from src.data_loader import DataLoader

from torch.utils.data import Dataset
import torch
import os

class Dataset_FFN(Dataset):
    def __init__(self, source_path, target_path, is_indexed=True, 
                 bpe_ops=7000, joined_bpe=False,
                 window_size=4, batch_alignment="uniform") -> None:
        """
        Initalizes the dataset for a FFN model from the specified source and target path. Data is separated into batches
        specified by the given arguments.

        Args:
            source_path (str): The path to the source data
            target_path (str): The path to the target data
            is_indexed (bool, optional): Whether the data is already indexed (i.e. no 
                BPE indexing through dictionary necessary). Defaults to True.
            bpe_ops (int, optional): The number of BPE operations to use. Defaults to 7000.
            joined_bpe (bool, optional): Whether to use joined BPE operations. Defaults to False.
            batch_size (int, optional): The size of the batches. Defaults to 200.
            window_size (int, optional): The window size for the batches. Defaults to 4.
            batch_alignment (str, optional): The alignment of the batches. Defaults to "uniform".
            model_type (str, optional): The type of model to use. Defaults to "feedforward".
        """
        
        self.batches = []
        self.batch_size = 1
        
        if is_indexed:
            source_indexed = PickleLoader.load(source_path)
            target_indexed = PickleLoader.load(target_path)
        else:
            source_indexed, target_indexed = self.__index_data(
                source_path, target_path, bpe_ops, joined_bpe)

        self.batcher = Batcher(source_indexed, target_indexed, 1, window_size, batch_alignment)
        self.batcher.batch()
        self.batches = self.batcher.getBatches()
        self.length = self.batcher.get_total_rows()
        self.joined_bpe = joined_bpe
        
        self.bpe_ops = bpe_ops

        self.source_dictionary = PickleLoader.load(f'../dictionaries/dict_{"JOINED_" if joined_bpe else ""}DE_{bpe_ops}.pkl')
        self.target_dictionary = PickleLoader.load(f'../dictionaries/dict_{"JOINED_" if joined_bpe else ""}EN_{bpe_ops}.pkl')
        self.source_bpe = PickleLoader.load(f"../encoder/BPE_{'JOINED' if joined_bpe else 'DE'}_{bpe_ops}.pickle")
        self.target_bpe = PickleLoader.load(f"../encoder/BPE_{'JOINED' if joined_bpe else 'EN'}_{bpe_ops}.pickle")


    def __index_data(self, source_path, target_path, bpe_ops, joined_bpe):
        """
        Private function to index the data using BPE and a dictionary.

        Args:
            source_path (str): The path to the source data
            target_path (str): The path to the target data
            bpe_ops (int): The number of BPE operations to use
            joined_bpe (bool): Whether to use joined BPE operations
        
        Returns:
            (tuple): A tuple containing the indexed source and target data
        """
        source_bpe, target_bpe = self.__get_encoder(source_path, target_path, bpe_ops, joined_bpe)
        source_dictionary, target_dictionary = self.__get_dictionary(source_path, target_path, bpe_ops, joined_bpe)

        source_data_loader = DataLoader(source_path)
        target_data_loader = DataLoader(target_path)

        source_data_encoded = source_bpe.encode_corpus(source_data_loader.load_data())
        target_data_encoded = target_bpe.encode_corpus(target_data_loader.load_data())

        source_data_indexed = source_dictionary.apply_mapping(source_data_encoded)
        target_data_indexed = target_dictionary.apply_mapping(target_data_encoded)

        # Store variables
        self.source_dictionary = source_dictionary
        self.target.dictionary = target_dictionary
        self.source_bpe = source_bpe
        self.target_bpe = target_bpe

        return (source_data_indexed, target_data_indexed)

    def __get_encoder(self, source_path, target_path, bpe_ops, joined_bpe):
        try:
            if joined_bpe:
                source_bpe = PickleLoader.load(f"../encoder/BPE_JOINED_{bpe_ops}.pickle")
                target_bpe = source_bpe
            else:
                source_bpe = PickleLoader.load(f"../encoder/BPE_DE_{bpe_ops}.pickle")
                target_bpe = PickleLoader.load(f"../encoder/BPE_EN_{bpe_ops}.pickle")
            return source_bpe, target_bpe
        except FileNotFoundError:
            if joined_bpe:
                source_data_loader = DataLoader(source_path)
                target_data_loader = DataLoader(target_path)

                merged_data = source_data_loader.load_data() + "\n" + target_data_loader.load_data()

                joined_bpe = BytePairEncoder()
                joined_bpe.fit(merged_data, operations=bpe_ops)
                PickleLoader.save(joined_bpe, f"../encoder/BPE_JOINED_{bpe_ops}.pickle")

                return joined_bpe, joined_bpe
            
            source_data_loader = DataLoader(source_path)
            source_bpe = BytePairEncoder()
            source_bpe.fit(source_data_loader.load_data(), operations=bpe_ops)
            PickleLoader.save(source_bpe, f"../encoder/BPE_DE_{bpe_ops}.pickle")

            target_data_loader = DataLoader(target_path)
            target_bpe = BytePairEncoder()
            target_bpe.fit(target_data_loader.load_data(), operations=bpe_ops)
            PickleLoader.save(target_bpe, f"../encoder/BPE_EN_{bpe_ops}.pickle")

            return source_bpe, target_bpe

    def __get_dictionary(self, source_path, target_path, bpe_ops, joined_bpe):
        source_dict_path = f"../dictionaries/dict_{'JOINED_' if joined_bpe else ''}DE_{bpe_ops}.pkl"
        target_dict_path = f"../dictionaries/dict_{'JOINED_' if joined_bpe else ''}EN_{bpe_ops}.pkl"
        try:
            source_dict = PickleLoader.load(source_dict_path)
            target_dict = PickleLoader.load(target_dict_path)
            return source_dict, target_dict
        except FileNotFoundError:
            source_data_loader = DataLoader(source_path)
            target_data_loader = DataLoader(target_path)

            if joined_bpe:
                source_bpe = PickleLoader.load(f"../encoder/BPE_JOINED_{bpe_ops}.pickle")
                target_bpe = source_bpe
            else:
                source_bpe = PickleLoader.load(f"../encoder/BPE_DE_{bpe_ops}.pickle")
                target_bpe = PickleLoader.load(f"../encoder/BPE_EN_{bpe_ops}.pickle")

            source_data_encoded = source_bpe.encode_corpus(source_data_loader.load_data())
            target_data_encoded = target_bpe.encode_corpus(target_data_loader.load_data())

            source_dict = Dictionary()
            source_dict.update(source_data_encoded)
            PickleLoader.save(source_dict_path, source_dict)

            target_dict = Dictionary()
            target_dict.update(target_data_encoded)
            PickleLoader.save(target_dict_path, target_dict)

            return source_dict, target_dict


    def __len__(self):
        """
        Returns the length of the dataset, i.e. the amount of training examples from the text data.

        Returns:
            int: The length of the dataset
        """
        return self.length

    def __getitem__(self, index):
        """
        
        """
        return (self.batches[index][0][0], 
                self.batches[index][1][0], 
                self.batches[index][2][0][0])
    
    def get_source_vocab_size(self):
        """
        Function to get the size of the source vocabulary.

        Returns:
            int: The size of the source vocabulary
        """
        return self.source_dictionary.get_vocab_size()

    def get_target_vocab_size(self):
        """
        Function to get the size of the target vocabulary.

        Returns:
            int: The size of the target vocabulary
        """
        return self.dictionary_get_vocab_size()


class Dataset_RNN(Dataset):
    def __init__(self, source_path, target_path, is_indexed=True,
                 bpe_ops=7000, joined_bpe=False):
        """
        Initializes the dataset for a RNN model from the specified source and target path.

        Args:
            source_path (str): The path to the source data
            target_path (str): The path to the target data
            is_indexed (bool, optional): Whether the data is already indexed (i.e. no
                BPE indexing through dictionary necessary). Defaults to True.
            bpe_ops (int, optional): The number of BPE operations to use. Defaults to 7000.
            joined_bpe (bool, optional): Whether to use joined BPE operations. Defaults to False.

        Raises:
            AssertionError: If the length of the source and target data is not equal
        """
        try:
            if is_indexed:
                source_indexed = PickleLoader.load(source_path)
                target_indexed = PickleLoader.load(target_path)
            else:
                source_indexed, target_indexed = self.__index_data(
                    source_path, target_path, bpe_ops, joined_bpe)
        except FileNotFoundError:
            source_indexed, target_indexed = self.__index_data(
                source_path, target_path, bpe_ops, joined_bpe)

        # same number of sentences
        assert len(source_indexed) == len(target_indexed)

        self.source = [torch.tensor(source) for source in source_indexed]
        # ignore SOS token?
        self.target = [torch.tensor(target[1:]) for target in target_indexed]

        source_dict_path = f"../dictionaries/dict_{'JOINED_' if joined_bpe else ''}DE_{bpe_ops}.pickle"
        target_dict_path = f"../dictionaries/dict_{'JOINED_' if joined_bpe else ''}EN_{bpe_ops}.pickle"

        self.source_dictionary = PickleLoader.load(source_dict_path)
        self.target_dictionary = PickleLoader.load(target_dict_path)
    
    def __index_data(self, source_path, target_path, bpe_ops, joined_bpe):
        """
        Private function to index the data using BPE and a dictionary.

        Args:
            source_path (str): The path to the source data
            target_path (str): The path to the target data
            bpe_ops (int): The number of BPE operations to use
            joined_bpe (bool): Whether to use joined BPE operations
        
        Returns:
            (tuple): A tuple containing the indexed source and target data
        """
        source_bpe, target_bpe = self.__get_encoder(source_path, target_path, bpe_ops, joined_bpe)
        source_dictionary, target_dictionary = self.__get_dictionary(source_path, target_path, bpe_ops, joined_bpe)

        source_data_loader = DataLoader(source_path)
        target_data_loader = DataLoader(target_path)

        source_data_encoded = source_bpe.encode_corpus(source_data_loader.load_data())
        target_data_encoded = target_bpe.encode_corpus(target_data_loader.load_data())

        source_data_indexed = source_dictionary.apply_mapping(source_data_encoded)
        target_data_indexed = target_dictionary.apply_mapping(target_data_encoded)

        # Store variables
        self.source_dictionary = source_dictionary
        self.target_dictionary = target_dictionary
        self.source_bpe = source_bpe
        self.target_bpe = target_bpe

        return (source_data_indexed, target_data_indexed)

    def __get_encoder(self, source_path, target_path, bpe_ops, joined_bpe):
        try:
            if joined_bpe:
                source_bpe = PickleLoader.load(f"../encoder/BPE_JOINED_{bpe_ops}.pickle")
                target_bpe = source_bpe
            else:
                source_bpe = PickleLoader.load(f"../encoder/BPE_DE_{bpe_ops}.pickle")
                target_bpe = PickleLoader.load(f"../encoder/BPE_EN_{bpe_ops}.pickle")
            return source_bpe, target_bpe
        except FileNotFoundError:
            if joined_bpe:
                source_data_loader = DataLoader(source_path)
                target_data_loader = DataLoader(target_path)

                merged_data = source_data_loader.load_data() + "\n" + target_data_loader.load_data()

                joined_bpe = BytePairEncoder()
                joined_bpe.fit(merged_data, operations=bpe_ops)
                PickleLoader.save(joined_bpe, f"../encoder/BPE_JOINED_{bpe_ops}.pickle")

                return joined_bpe, joined_bpe
            
            source_data_loader = DataLoader(source_path)
            source_bpe = BytePairEncoder()
            source_bpe.fit(source_data_loader.load_data(), operations=bpe_ops)
            PickleLoader.save(source_bpe, f"../encoder/BPE_DE_{bpe_ops}.pickle")

            target_data_loader = DataLoader(target_path)
            target_bpe = BytePairEncoder()
            target_bpe.fit(target_data_loader.load_data(), operations=bpe_ops)
            PickleLoader.save(target_bpe, f"../encoder/BPE_EN_{bpe_ops}.pickle")

            return source_bpe, target_bpe

    def __get_dictionary(self, source_path, target_path, bpe_ops, joined_bpe):
        source_dict_path = f"../dictionaries/dict_{'JOINED_' if joined_bpe else ''}DE_{bpe_ops}.pkl"
        target_dict_path = f"../dictionaries/dict_{'JOINED_' if joined_bpe else ''}EN_{bpe_ops}.pkl"
        try:
            source_dict = PickleLoader.load(source_dict_path)
            target_dict = PickleLoader.load(target_dict_path)
            return source_dict, target_dict
        except FileNotFoundError:
            source_data_loader = DataLoader(source_path)
            target_data_loader = DataLoader(target_path)

            if joined_bpe:
                source_bpe = PickleLoader.load(f"../encoder/BPE_JOINED_{bpe_ops}.pickle")
                target_bpe = source_bpe
            else:
                source_bpe = PickleLoader.load(f"../encoder/BPE_DE_{bpe_ops}.pickle")
                target_bpe = PickleLoader.load(f"../encoder/BPE_EN_{bpe_ops}.pickle")

            source_data_encoded = source_bpe.encode_corpus(source_data_loader.load_data())
            target_data_encoded = target_bpe.encode_corpus(target_data_loader.load_data())

            source_dict = Dictionary()
            source_dict.update(source_data_encoded)
            PickleLoader.save(source_dict_path, source_dict)

            target_dict = Dictionary()
            target_dict.update(target_data_encoded)
            PickleLoader.save(target_dict_path, target_dict)

            return source_dict, target_dict

    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]

    def get_source_vocab_size(self):
        """
        Function to get the size of the source vocabulary.
        
        Returns:
        int: The size of the source vocabulary
        """
        return self.source_dictionary.get_vocab_size()

    def get_target_vocab_size(self):
        """
        Function to get the size of the target vocabulary.

        Returns:
            int: The size of the target vocabulary
        """
        return self.target_dictionary.get_vocab_size()

    def get_source_dictionary(self):
        """
        Function to get the source dictionary.

        Returns:
            Dictionary: The source dictionary
        """
        return self.source_dictionary

    def get_target_dictionary(self):
        """
        Function to get the target dictionary.

        Returns:
            Dictionary: The target dictionary
        """
        return self.target_dictionary
    

if __name__ == "__main__":
    dataset = Dataset_FFN("train/multi30k.de.gz", 
                        "train/multi30k.en.gz",
                        is_indexed=False)
    
    print(len(dataset))
    print(dataset[0])
