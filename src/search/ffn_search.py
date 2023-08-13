import sys
sys.path.append('..')

from src.preprocessing.dictionary import START, END
from src.models.ff_model import FFN
from src.utils.get_list_item_safe import get_list_item_safe
from src.utils.alignment import uniform_alignment
from src.pickle_loader import PickleLoader

import torch
import subprocess
from torch.nn.functional import log_softmax
from torch import topk
import torch


def beam_search(source, model, beam_size, max_decoding_time_step, source_target_ratio, alignment_modeling="average", temperature=1):
    """
    Beam search algorithm

    Parameters:
        src (List[int]): Source sentence (indexed)
        model (Model): Model to use for decoding
        beam_size (int): Beam size
        max_decoding_time_step (int): Maximum number of decoding steps
        alignment_modeling (str): Alignment modeling method. Currently only "average" is supported
    
    Returns:
        List[List[int]]: List of decoded sentences (indexed)
    """
    model.eval()
    # get average source target ratio
    # universal_newlines so that new line character is always \n
    #output = subprocess.check_output(['bash', '../scripts/source_target_ratio.sh', "../data/unpreprocessed/train/source.train" , "../data/unpreprocessed/train/target.train"], universal_newlines=True)
    #source_target_ratio = float(output)
    #Load the indexed train data with pickle loader
    if alignment_modeling == "average":
        # Estimate target length
        estimated_target_length = round(len(source) / source_target_ratio)
        # get alignment
        align = uniform_alignment(len(source), estimated_target_length)
    else:
        raise ValueError("Alignment modeling method not supported")
    
    # get window size
    window_size = model.config['window_size']

    # initialize beam
    beam = [([START], 0)]

    # loop over 
    # loop over decoding time steps
    for t in range(max_decoding_time_step):
        # initialize candidates
        candidates = []
        # loop over beam
        for sentence, score in beam:
            # if sentence ends with END, add to candidates
            if sentence[-1] == END:
                candidates.append((sentence, score))
                continue
            
            # unnormalize score

            n = len(sentence)
            score = score * n

            # get source window
            source_window = torch.tensor([get_list_item_safe(source, align(i)) for i in range(n - window_size, n + window_size+1)])
            # Add one dimension to make it a batch of size 1
            source_window = source_window.unsqueeze(0)
            # get target window
            target_window = torch.tensor([get_list_item_safe(sentence, i) for i in range(n - window_size, n)])
            # Add one dimension to make it a batch of size 1
            target_window = target_window.unsqueeze(0)
            # get logits
            logits = model(source_window, target_window)
            if temperature != 1 and temperature > 0:
                logits = logits / temperature
            # get probability distribution
            prob_dist = log_softmax(logits, dim=1)
            # remove batch dimension
            prob_dist = prob_dist.squeeze(0)
            # get top k candidates
            top_k = topk(input=prob_dist, k=beam_size)
            # loop over top k candidates
            for i in range(beam_size):
                # get index and normalized score
                candidate_index = top_k.indices[i].item()
                # Calculate new score and normalize
                candidate_score = (top_k.values[i].item() + score) / (n + 1)
                # add candidate to candidates
                candidates.append((sentence + [candidate_index], candidate_score))

        # sort candidates by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        # set beam to top k candidates
        beam = candidates[:beam_size]
        
    # return decoded sentences
    return [sentence for sentence, score in beam]


def greedy_search(source, model, max_decoding_time_step, source_target_ratio):
    """
    Greedy search algorithm

    Parameters:
        src (List[int]): Source sentence (indexed)
        model (Model): Model to use for decoding
        max_decoding_time_step (int): Maximum number of decoding steps
    
    Returns:
        List[int]: Decoded sentence (indexed)

    """
    return beam_search(source=source, model=model, beam_size=1, 
                       max_decoding_time_step=max_decoding_time_step, source_target_ratio=source_target_ratio)[0]


if __name__ == "__main__":
    MODEL_PATH = "../logs/gutere_modell/checkpoints/model_6.pt"
    model = torch.load(MODEL_PATH)

    source = [START, 2, 3, 4, 5, 6, 7, 8, END]

    print(source)
    print(type(source))
    print(beam_search(source, model, 3, 20))