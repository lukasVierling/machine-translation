import sys
sys.path.append("..")
import src.utils.alignment as alignment
from src.utils.get_list_item_safe import get_list_item_safe

from typing import List, Callable, Tuple, Any, Union
import torch
from tqdm import tqdm


class Batcher():
    """
    A class for batching int-tokenized source and target sentences.

    Usage:
        batcher = Batcher(source, target, batch_size=200, window=4, alignment="uniform")
        batcher.batch()
        do_something(batcher.getBatches())

    Methods:
        batch(): Creates batches, specified by source, target, batch_size, window and alignment and saves them
            in the batches attribute
        getBatches(): Returns the batches created by batch() method or [] if batch() has not been called yet

    """
    def __init__(self, source: List[List[int]], target: List[List[int]], 
                 batch_size: int = 200, window: int = 4, alignment: Union[str, Callable[[int], int]] = "uniform", 
                 torch_device = None) -> None:
        """
        Iniates a Batcher object for batching of source and target sentences.

        Args:
            source (List[List[int]]): List of source sentences, represented as int-tokenized sentence
            target (List[List[int]]): List of target sentences, represented as strings
            batch_size (int): The maximum size of each batch. Defaults to 200.
            window (int): The number of sentences to consider for each batch. Defaults to 4.
            alignment (str): The alignment strategy to use when creating batches. Defaults to "uniform".
                Can be choosen from "uniform", "stepwise"
            torch_device (str): The device to use for the batches. Defaults to None. 
                Can be choosen from "cpu", "cuda" (Nvidia GPU), "mps" (Apple Silicon) and others 
                (for more info see https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device)
        """
        
        assert len(source) == len(target), f"Error: Source and target must have equal amount of sentences. Source: {len(source)}, Target: {len(target)} "

        self.__source, self.__target = self.__filter_empty_sentences(source, target)

        self.__batch_size = batch_size
        self.__window = window

        self.__alignment = alignment
        self.__torch_device = torch_device

        self.__batches = []

    def __filter_empty_sentences(self, source: List[List[int]], target: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Private function to filter out empty sentences from a list of sentences. A sentence is considered
        empty if it has 2 items (start-of-sentence and end-of-sentence token). 
        Both the source and corresponding target sentence are deleted, if one of them is empty.
        
        Args:
            source (List[List[int]]): List of source sentences to filter
            target (List[List[int]]): List of target sentences to filter

        Returns:
            Tuple[List[List[int]], List[List[int]]]: Source and target list of sentences without empty sentences
        """
        filtered_source = []
        filtered_target = []

        for src, tgt in zip(source, target):
            if len(src) > 2 and len(tgt) > 2:
                filtered_source.append(src)
                filtered_target.append(tgt)

        return filtered_source, filtered_target

    def __align(self, source: List[int], target: List[int]) -> Callable[[int], int]:
        """
        Private function to define an alignment function between the specified
        int-tokenized source and targets sentence.

        Args:
            source (List[int]): Source sentence to align on
            target (List[int]): Target sentence to align from

        Returns:
            Callable[[int], int]: A callable function that takes an integer index
            in the target sentence and returns the corresponding index in the
            source sentence for alignment.
        """
        if self.__alignment == "uniform":
            return alignment.uniform_alignment(len(source), len(target))
        if self.__alignment == "stepwise":
            return alignment.stepwise_alignment(len(source), len(target))
        if callable(self.__alignment):
            return self.__alignment
    

    def __get_list_item_safe(self, ls: List[Any], index: int) -> Any:
        """
        Private function to get an item from a list at a specified index. 
        If the index is out of bounds of the list, the first or last item
        of the list respectively is returned.

        Args:
            ls (List[Any]): List to get item from
            index (int): Index of item to get
        
        Returns:
            Any: Item at specified index or first/last item if index is out of bounds
        """
        return get_list_item_safe(ls, index)

    def __initialize_batches(self, total_rows: int) -> None:
        """
        Private function for initializing batches of data for a machine translation model

        Args:
            total_rows (int): Total number of rows which have to be divided into batches
        """

        # create full batches, of which total_rows//batch_size exist
        self.__batches = [(
            torch.zeros(size=(self.__batch_size, self.__window * 2 +1), dtype=torch.long, device=self.__torch_device),
            torch.zeros(size=(self.__batch_size, self.__window), dtype=torch.long, device=self.__torch_device),
            torch.zeros(size=(self.__batch_size, 1), dtype=torch.long, device=self.__torch_device)
        ) for _ in range(total_rows//self.__batch_size)]

        # create partial batch, which has remainding rows as size (total_rows % batch_size)
        self.__batches.append((
            torch.zeros(size=(total_rows % self.__batch_size, self.__window * 2 + 1), dtype=torch.long, device=self.__torch_device),
            torch.zeros(size=(total_rows % self.__batch_size, self.__window), dtype=torch.long, device=self.__torch_device),
            torch.zeros(size=(total_rows % self.__batch_size, 1), dtype=torch.long, device=self.__torch_device)
        ))

    def get_total_rows(self) -> int:
        """
        Calculates the total number of rows that the batches will have. This is the total
        number of words in the target sentences minus the number of start-of-sentence tokens.

        Returns:
            int: Total number of rows that the batches will have
        """
        return len([word for sentence in self.__target for word in sentence[1:]])   

    def batch(self) -> None:
        """
        Performs batching on the given source and target sentences with the specified window
        and batch sizes.
        """
        total_rows = self.get_total_rows()
        self.__initialize_batches(total_rows)

        curr_batch = 0
        batch_row = 0

        progress_bar = tqdm(zip(self.__source, self.__target), total=len(self.__source), desc="Creating batches")
        for (source_sentence, target_sentence) in progress_bar:
            align = self.__align(self.__source, self.__target)
            # start at first word, since SOS is not included in batches
            for (target_index, target_word) in enumerate(target_sentence[1:]):
                # ensure prediction begins after start-of-sentence token
                # if target_index == 0:
                #     continue

                source_index = align(target_index)

                S, T, L = self.__batches[curr_batch]

                # fill source window tensor
                window_offset = self.__window * -1
                for batch_col in range(self.__window * 2 + 1):
                    S[batch_row, batch_col] = self.__get_list_item_safe(source_sentence, source_index + window_offset)
                    window_offset += 1

                window_offset = self.__window * -1 + 1
                # fill target window tensor
                for batch_col in range(self.__window):
                    T[batch_row, batch_col] = self.__get_list_item_safe(target_sentence, target_index + window_offset)
                    window_offset += 1

                # fill target label tensor
                L[batch_row] = target_word

                postfix_str = {"Batch": f"{curr_batch + 1}/{len(self.__batches)}", 
                               "Row": f"{batch_row + 1}/{self.__batch_size}" if curr_batch < len(self.__batches)
                                else f"{batch_row + 1}/{total_rows % self.__batch_size}"}
                progress_bar.set_postfix(postfix_str, refresh=False)

                # update batch indices
                batch_row += 1
                if batch_row == self.__batch_size:
                    batch_row = 0
                    curr_batch += 1

    def getBatches(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Returns the batches created after the batch() function has been called.
        If batch() has not been called yet, empty list [] is returned.
        
        Returns:
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: List of the batches created by batch() function
                Each batch is a tuple (S, T, L) where
                    - S is the source windows matrix with size (B x (w*2 + 1))
                    - T is the target windows matix with size (B x w)
                    - L is the target labels vector with size (B)
                Consider that the last batch might have smaller size than B
        """
        return self.__batches

if __name__ == "__main__":
    batcher = Batcher([], [], batch_size=2, window=4, alignment="uniform")

    """
    batcher.batch()
    print(batcher.batches)

    test = [
        [1, 2, 3], 
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2]]
    """
    for i in range(3, 100):
        for j in range(3, 100):
            source = j * [0]
            target = i * [0]
            align = batcher.__align(source, target)

            assert align(i-1) == j-1, f"Failed at i = {i}, j = {j}. \nAlign({i-1}) = {align(i-1)}. Expected: {j-1}\ntarget: {target} source: {source}"
