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

def beam_search(source, model, beam_size, max_decoding_time_step, source_target_ratio=0, alignment_modeling=None, temperature=1):
    """
    Beam search algorithm, extra parameters to make it work in the decode script

    Parameters:
        src (List[int]): Source sentence (indexed)
        model (Model): Model to use for decoding
        beam_size (int): Beam size
        max_decoding_time_step (int): Maximum number of decoding steps
        alignment_modeling (str): Alignment modeling method. Currently only "average" is supported
    
    Returns:
        List[List[int]]: List of decoded sentences (indexed)
    """
    encoder, decoder = model
    encoder.eval()
    decoder.eval()
    # get average source target ratio
    # universal_newlines so that new line character is always \n
    #output = subprocess.check_output(['bash', '../scripts/source_target_ratio.sh', "../data/unpreprocessed/train/source.train" , "../data/unpreprocessed/train/target.train"], universal_newlines=True)
    #source_target_ratio = float(output)
    #Load the indexed train data with pickle loader

    # Get source length
    source_len = len(source)
    # Create batch for source
    source = torch.tensor(source).unsqueeze(0)
    # Get encoded source
    encoder_output, _ = encoder(source,[source_len])

    # Initial prev_internal_state ( batch size on pos 1)
    if decoder.rnn_type == 'lstm':
        prev_hidden = torch.zeros(decoder.num_rnn_layers, 1, decoder.decoder_rnn_hidden_size)
        prev_cell_state = torch.zeros(decoder.num_rnn_layers, 1, decoder.decoder_rnn_hidden_size)
        prev_internal_state = (prev_hidden, prev_cell_state)
    else:
        prev_internal_state = torch.zeros(decoder.num_rnn_layers, 1, decoder.decoder_rnn_hidden_size)
    mask=torch.zeros(1,source_len)
    
    # initialize beam
    beam = [([START], 0, prev_internal_state)]
    # loop over decoding time steps
    for t in range(max_decoding_time_step):
        # initialize candidates
        candidates = []
        # loop over beam
        for sentence, score, prev_internal_state in beam:
            # if sentence ends with END, add to candidates
            if sentence[-1] == END:
                candidates.append((sentence, score, prev_internal_state))
                continue
            
            # unnormalize score

            n = len(sentence)
            score = score * n
            # get next input token
            input_token = sentence[-1]
            # add batch dimension
            input_token = torch.tensor([input_token]).unsqueeze(0)
            # get logits
            logits, prev_internal_state = decoder.forward_step(input_token, encoder_output, prev_internal_state, source_len, mask)

            # Temperature
            if temperature != 1 and temperature > 0:
                logits = logits / temperature

            # get probability distribution
            logits = logits.squeeze(0).squeeze(0)
            prob_dist = log_softmax(logits, dim=0)
            # remove batch dimension
            
            # get top k candidates
            top_k = topk(input=prob_dist, k=beam_size)
            # loop over top k candidates
            for i in range(beam_size):
                # get index and normalized score
                candidate_index = top_k.indices[i].item()
                # Calculate new score and normalize
                candidate_score = (top_k.values[i].item() + score) / (n + 1)
                # add candidate to candidates
                candidates.append((sentence + [candidate_index], candidate_score, prev_internal_state))

        # sort candidates by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        # set beam to top k candidates
        beam = candidates[:beam_size]
        
    # return decoded sentences
    return [sentence for sentence, _, _ in beam]

def greedy_search(source, model, max_decoding_time_step):
    raise NotImplementedError("Greedy search not implemented yet")
