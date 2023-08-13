import sys
sys.path.append('..')

import argparse
import torch
import os

from src.pickle_loader import PickleLoader
from src.search.rnn_search import beam_search as rnn_beam_search
from src.utils.byte_pair_encoding import BytePairEncoder
from src.preprocessing.dictionary import Dictionary

import torch

def decode_str(model_path,top_k,beam_size,max_decoding_time_step,sentence,temperature=1):

    model = torch.load(model_path, map_location=torch.device('cpu'))

    encoder, decoder = model
    config = encoder.config
    beam_search = rnn_beam_search
    
    # load encoder
    source_encoder_path = f"../../encoder/BPE_{'JOINED' if config['joined_bpe'] else 'DE'}_{config['bpe_ops']}.pickle"
    source_encoder = PickleLoader.load(source_encoder_path)
    target_encoder_path = f"../../encoder/BPE_{'JOINED' if config['joined_bpe'] else 'EN'}_{config['bpe_ops']}.pickle"
    target_encoder = PickleLoader.load(target_encoder_path)

    #calc the ratio
    train_source_data = PickleLoader.load(f"../../data/{config['bpe_ops']}_BPE_indexed/train/source.train.pickle")
    train_target_data = PickleLoader.load(f"../../data/{config['bpe_ops']}_BPE_indexed/train/target.train.pickle")
    # get average sentence length for source and target
    source_average_sentence_len = sum([len(sentence) for sentence in train_source_data]) / len(train_source_data)
    target_average_sentence_len = sum([len(sentence) for sentence in train_target_data]) / len(train_target_data)
    source_target_ratio = source_average_sentence_len / target_average_sentence_len
    #print(source_target_ratio)
    # load dictionary
    source_dictionary_path = f"../../dictionaries/dict_{'JOINED_' if config['joined_bpe'] else ''}DE_{config['bpe_ops']}.pickle"
    source_dictionary = PickleLoader.load(source_dictionary_path)

    target_dictionary_path = f"../../dictionaries/dict_{'JOINED_' if config['joined_bpe'] else ''}EN_{config['bpe_ops']}.pickle"
    target_dictionary = PickleLoader.load(target_dictionary_path)

    # encode corpus
    source_encoded = source_encoder.encode_corpus(sentence)
    #target_encoded = target_encoder.encode_corpus(loader.load_data())

    # indexed corpus
    source_indexed = source_dictionary.apply_mapping(source_encoded)
    #target_indexed = target_dictionary.apply_mapping(target_encoded)
    assert type(source_indexed) == list and type(source_indexed[0]) == list and type(source_indexed[0][0]) == int, "Source corpus is not indexed"

    n_best = min(top_k, beam_size) if top_k > 0 else beam_size

    # loop over corpus
    for indexed_line in source_indexed:
        # beam search and choose n_best
        translations = beam_search(source=indexed_line, model=model, beam_size=beam_size, 
                                    max_decoding_time_step=max_decoding_time_step, 
                                    alignment_modeling="average", temperature=temperature, source_target_ratio=source_target_ratio)[:n_best]
        # remove mapping
        translations = target_dictionary.remove_mapping(translations)
        # remove bpe optionally
        translations = [target_encoder.decode(translation) for translation in translations]
    
    #flatten into one string and add . between the sentences
    translations = ". ".join(translations)

    return translations