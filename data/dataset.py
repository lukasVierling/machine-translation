import sys
sys.path.append('..')

from src.preprocessing.batching import Batcher
from src.preprocessing.dictionary import Dictionary
from src.utils.byte_pair_encoding import BytePairEncoder
from src.pickle_loader import PickleLoader
from src.data_loader import DataLoader

from torch.utils.data import Dataset

class MTDataset(Dataset):
    def __init__(self, source_path, target_path, is_indexed=True, 
                 bpe_ops=7000, joined_bpe=False,
                 window_size=4, batch_alignment="uniform") -> None:
        """
        Initalizes the dataset from the specified source and target path. Data is separated into batches
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
        # currently only 7000 BPE joined operations supported
        if bpe_ops == 7000 and not joined_bpe:
            source_bpe = PickleLoader.load("../encoder/BPE_DE_7000.pickle")
            target_bpe = PickleLoader.load("../encoder/BPE_EN_7000.pickle")

            source_dictionary = PickleLoader.load("../dictionaries/dict_DE_7000.pkl")
            target_dictionary = PickleLoader.load("../dictionaries/dict_EN_7000.pkl")
        else:
            raise Exception("Only unjoined 7000 BPE operations supported at the moment")

        source_data_loader = DataLoader(source_path)
        target_data_loader = DataLoader(target_path)

        source_data_encoded = source_bpe.encode_corpus(source_data_loader.load_data())
        target_data_encoded = target_bpe.encode_corpus(target_data_loader.load_data())

        source_data_indexed = source_dictionary.apply_mapping(source_data_encoded)
        target_data_indexed = target_dictionary.apply_mapping(target_data_encoded)

        return (source_data_indexed, target_data_indexed)

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
        if self.joined_bpe:
            dictionary = PickleLoader.load(f"../dictionaries/dict_JOINED_DE_{self.bpe_ops}.pkl")
        else:
            dictionary = PickleLoader.load(f"../dictionaries/dict_DE_{self.bpe_ops}.pkl")
        return dictionary.get_vocab_size()

    def get_target_vocab_size(self):
        """
        Function to get the size of the target vocabulary.

        Returns:
            int: The size of the target vocabulary
        """
        if self.joined_bpe:
            dictionary = PickleLoader.load(f"../dictionaries/dict_JOINED_DE_{self.bpe_ops}.pkl")
        else:
            dictionary = PickleLoader.load(f"../dictionaries/dict_EN_{self.bpe_ops}.pkl")
        return dictionary.get_vocab_size()
    

if __name__ == "__main__":
    dataset = MTDataset("train/multi30k.de.gz", 
                        "train/multi30k.en.gz",
                        is_indexed=False)
    
    print(len(dataset))
    print(dataset[0])