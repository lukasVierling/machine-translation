#Script that is called with the path to a folder. In the folder are an arbitrary amount of pytroch mdoel checkpoints.
#The script then evaluates the BLEU score of each model and prints the results to the console.
#The script also prints the best model and its BLEU score to the console.
import sys
sys.path.append('..')

import torch
import os
from torch import nn
from tqdm import tqdm
import argparse
import json
import pathlib

from src.pickle_loader import PickleLoader
from data.dataset import Dataset_FFN
from src.models import FFN
from src.search.ffn_search import beam_search as beam_search_ffn
from src.search.rnn_search import beam_search as beam_search_rnn
import src
from src.utils.metrics import Metrics

def BLEU_evaluate_checkpoints_FFN(path_to_folder, beam_size, max_decoding_time_step,temperature=1):
    """
    Evaluate all models in a folder and print the results to the console.

    Args:
        path_to_folder (str): The path to the folder containing the models.
        beam_size (int): The size of the beam for beam search decoding.
        max_decoding_time_step (int): The maximum decoding time step.
    """

    # Get all files in the folder
    files = os.listdir(path_to_folder)
    # Create a dictionary to store the models and their BLEU scores
    models = {}
    # Loop over all files
    progress_bar = tqdm(files)
    for file in progress_bar:
        print(file)
        # Load the model
        #check if file ends with .pt
        if not(file.endswith('.pt')):
            continue
        model = torch.load(os.path.join(path_to_folder, file))
        # Check if the model is a pytorch model
        if not isinstance(model, torch.nn.Module):
            continue
        
        #extract the parameters for the bleu eval function from the model hyperparameters
        model_config = model.config
        bpe_ops = model_config['bpe_ops']
        joined_bpe = model_config['joined_bpe']
        alignment_modeling = 'average'
        dev_source  = '../data/unpreprocessed/dev/source.dev'
        dev_target = '../data/unpreprocessed/dev/target.dev'
        bleu_score = BLEU_eval_FFN(model=model, 
                               bleu_n=4,
                               dev_source=dev_source,
                               dev_target=dev_target,
                               bpe_ops=bpe_ops,
                               joined_bpe=joined_bpe,
                               alignment_modeling=alignment_modeling,
                               beam_size=beam_size,
                               max_decoding_time_step=max_decoding_time_step,
                               temperature=temperature
                               )
        # Store the model and its BLEU score
        print('finished calculating BLEU score')
        print(bleu_score)
        models[file] = bleu_score
    # Print the results
    for model in models:
        print(model + ": " + str(models[model]))
    # Print the best model
    best_model = max(models, key=models.get)
    print("Best model: " + best_model + " with a BLEU score of " + str(models[best_model]))
    return models

def BLEU_eval_FFN(model, bleu_n, dev_source, dev_target, bpe_ops, 
              joined_bpe, beam_size, max_decoding_time_step, 
              alignment_modeling='average', temperature=1):
    """
    Evaluate the BLEU score of a model.

    Args:
        model (torch.nn.module): The model to evaluate.
        bleu_n (int): The value of n for BLEU score calculation.
        dev_source (str): The path to raw source data for evaluation.
        dev_target (str): The path to target data for evaluation.
        bpe_ops (int): The BPE operations for tokenization.
        joined_bpe (bool): Whether the BPE tokens are joined or separated.
        beam_size (int): The size of the beam for beam search decoding.
        max_decoding_time_step (int): The maximum decoding time step.
        alignment_modeling (str): The alignment modeling strategy. Defaults to 'average'.
        temperature (float): The temperature for the softmax function. Defaults to 1.

    Returns:
        float: The BLEU score of the model on the evaluation data.
    """
    # Create Metrics object
    metric = Metrics()
    if not(joined_bpe):
        source_dictionary = PickleLoader.load(f"../dictionaries/dict_DE_{bpe_ops}.pickle")
        target_dictionary = PickleLoader.load(f"../dictionaries/dict_EN_{bpe_ops}.pickle")
        source_bpe = PickleLoader.load(f"../encoder/BPE_DE_{bpe_ops}.pickle")
        target_bpe = PickleLoader.load(f"../encoder/BPE_EN_{bpe_ops}.pickle")
    else:
        source_dictionary = PickleLoader.load(f"../dictionaries/dict_JOINED_DE_{bpe_ops}.pickle")
        target_dictionary = PickleLoader.load(f"../dictionaries/dict_JOINED_EN_{bpe_ops}.pickle")
        source_bpe = PickleLoader.load(f"../encoder/BPE_JOINED_{bpe_ops}.pickle")
        target_bpe = PickleLoader.load(f"../encoder/BPE_JOINED_{bpe_ops}.pickle")
    # Create validation dataset and loader
    dev_source_dl = src.DataLoader(dev_source)
    dev_target_dl = src.DataLoader(dev_target)
    # Index dev set
    encoded_corpus = source_bpe.encode_corpus(dev_source_dl.load_data())
    indexed_corpus = source_dictionary.apply_mapping(encoded_corpus) 

    #calc the ratio
    train_source_data = PickleLoader.load(f"../data/{model.config['bpe_ops']//1000}k_BPE_indexed/train/source.train.pickle")
    train_target_data = PickleLoader.load(f"../data/{model.config['bpe_ops']//1000}k_BPE_indexed/train/target.train.pickle")
    # get average sentence length for source and target
    source_average_sentence_len = sum([len(sentence) for sentence in train_source_data]) / len(train_source_data)
    target_average_sentence_len = sum([len(sentence) for sentence in train_target_data]) / len(train_target_data)
    source_target_ratio = source_average_sentence_len / target_average_sentence_len

    # Evaluation mode
    model.eval()
    with torch.no_grad():
        bleu_score = 0
        hypothesis = []
        for sentence in indexed_corpus:
            # Calcualte the hypothesis
            hypothesis += [beam_search_ffn(source=sentence, 
                                       model=model, 
                                       beam_size=beam_size,
                                       max_decoding_time_step=max_decoding_time_step, source_target_ratio=source_target_ratio,
                                       alignment_modeling=alignment_modeling,temperature=temperature)[0]]

    # Decode the text
    hypothesis_bpe = target_dictionary.remove_mapping(hypothesis)
    hypothesis_str = target_bpe.decode_corpus(hypothesis_bpe)
    #createa a file in the results folder and write the hypthoesis into the file as txt file
    # create the folder if it does not exist
    if not os.path.exists(f"../results/{model.config['bpe_ops']//1000}k_BPE_indexed"):
        os.makedirs(f"../results/{model.config['bpe_ops']//1000}k_BPE_indexed")
    with open(f"../results/{model.config['bpe_ops']//1000}k_BPE_indexed/hyp.txt", "w") as file:
        for line in hypothesis_str:
            file.write(line + "\n")

    # Calculate the BLEU score
    bleu_score = metric.bleu_score(bleu_n,hypothesis_str,dev_target_dl.tokenize('lines'))
    return bleu_score

def BLEU_evaluate_checkpoints_RNN(path_to_folder, beam_size, max_decoding_time_step,temperature=1):
    """
    Evaluate all RNN models in a folder and print the results to the console.

    Args:
        path_to_folder (str): The path to the folder containing the models.
        beam_size (int): The size of the beam for beam search decoding.
        max_decoding_time_step (int): The maximum decoding time step.
        temperature (float): The temperature for the softmax function. Defaults to 1.
    
    Returns:
        dict: A dictionary containing the model names and their BLEU scores.
    """

    # Get all files in the folder
    files = os.listdir(path_to_folder)
    # Create a dictionary to store the models and their BLEU scores
    models = {}
    # Loop over all files
    progress_bar = tqdm(files)
    for file in progress_bar:
        print(file)
        # Load the model
        #check if file ends with .pt
        if not(file.endswith('.pt')):
            continue
        model = torch.load(os.path.join(path_to_folder, file), map_location=torch.device('cpu'))
        # Check if the model is a pytorch model
        is_encoder_decoder = isinstance(model, tuple) and len(model) == 2 and isinstance(model[0], torch.nn.Module) and isinstance(model[1], torch.nn.Module)
        if not is_encoder_decoder:
            continue
        
        #extract the parameters for the bleu eval function from the model hyperparameters
        encoder, decoder = model
        model_config = encoder.config
        bpe_ops = model_config['bpe_ops']
        joined_bpe = model_config['joined_bpe']
        dev_source  = '../data/raw/dev/source.dev'
        dev_target = '../data/raw/dev/target.dev'
        print('Calculating BLEU score...')
        bleu_score = BLEU_eval_RNN(encoder=encoder,
                                   decoder=decoder,
                                   bleu_n=4,
                                   dev_source=dev_source,
                                   dev_target=dev_target,
                                   bpe_ops=bpe_ops,
                                   joined_bpe=joined_bpe,
                                   beam_size=beam_size,
                                   max_decoding_time_step=max_decoding_time_step,
                                   temperature=temperature)

        # Store the model and its BLEU score
        print(bleu_score)
        models[file] = bleu_score
        print("")

    print('\nfinished calculating BLEU scores') 
    # Print the results
    for model in models:
        print(model + ": " + str(models[model]))
    # Print the best model
    best_model = max(models, key=models.get)
    print("Best model: " + best_model + " with a BLEU score of " + str(models[best_model]))
    
    return models

def BLEU_eval_RNN(encoder, decoder, bleu_n, dev_source, dev_target, bpe_ops, 
              joined_bpe, beam_size, max_decoding_time_step, temperature=1):         
    """
    Evaluate the BLEU score of a RNN model.

    Args:
        encoder (torch.nn.module): The encoder of the model to evaluate.
        decoder (torch.nn.module): The decoder of the model to evaluate.
        bleu_n (int): The value of n for BLEU score calculation.
        dev_source (str): The path to raw source data for evaluation.
        dev_target (str): The path to target data for evaluation.
        bpe_ops (int): The BPE operations for tokenization.
        joined_bpe (bool): Whether the BPE tokens are joined or separated.
        beam_size (int): The size of the beam for beam search decoding.
        max_decoding_time_step (int): The maximum decoding time step.
        temperature (float): The temperature for the softmax function. Defaults to 1.
    
    Returns:
        float: The BLEU score of the model on the evaluation data.
    """
    # Create Metrics object
    metric = Metrics()
    source_dictionary = PickleLoader.load(f"../dictionaries/dict_{'JOINED_' if joined_bpe else ''}DE_{bpe_ops}.pickle")
    target_dictionary = PickleLoader.load(f"../dictionaries/dict_{'JOINED_' if joined_bpe else ''}EN_{bpe_ops}.pickle")
    source_bpe = PickleLoader.load(f"../encoder/BPE_{'JOINED' if joined_bpe else 'DE'}_{bpe_ops}.pickle")
    target_bpe = PickleLoader.load(f"../encoder/BPE_{'JOINED' if joined_bpe else 'EN'}_{bpe_ops}.pickle")
    
    # Create validation dataset and loader
    dev_source_dl = src.DataLoader(dev_source)
    dev_target_dl = src.DataLoader(dev_target)
    # Index dev set
    encoded_corpus = source_bpe.encode_corpus(dev_source_dl.load_data())
    indexed_corpus = source_dictionary.apply_mapping(encoded_corpus) 

    # Evaluation mode
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        bleu_score = 0
        hypothesis = []
        for sentence in indexed_corpus:
            # Calcualte the hypothesis
            hypothesis += [beam_search_rnn(source=sentence, 
                                       model=(encoder, decoder), 
                                       beam_size=beam_size,
                                       max_decoding_time_step=max_decoding_time_step,
                                       temperature=temperature)[0]]

    # Decode the text
    hypothesis_bpe = target_dictionary.remove_mapping(hypothesis)
    hypothesis_str = target_bpe.decode_corpus(hypothesis_bpe)

    # Calculate the BLEU score
    bleu_score = metric.bleu_score(bleu_n, hypothesis_str, dev_target_dl.tokenize('lines'))
    return bleu_score
             
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_folder', type=str, required=True, help='The path to the folder containing the models.')
    parser.add_argument('--beam_size', type=int, default=20, help='The size of the beam for beam search decoding.')
    parser.add_argument('--max_decoding_time_step', type=int, default=100, help='The maximum decoding time step.')
    parser.add_argument('--temperature', type=float, default=1.0, help='The temperature for the softmax function. Defaults to 1.')
    # model type argument, either FFN or RNN
    parser.add_argument('--model_type', choices=['FFN', 'RNN'], required=True, help='The type of model to evaluate.')
    parser.add_argument('--dest_path', type=str, required=True, help='The path of the file to save the results in.')
    parser.add_argument('--use_pickle', action='store_true', help='Flag indicating if the results should be saved as a pickle file. Otherwise, the results are saved as a text file')
    parser.add_argument('--source_path', type=str, help='The path to the source data (raw) for evaluation.')
    args = parser.parse_args()

    # create path if necessary
    pathlib.Path(args.dest_path).parent.mkdir(parents=True, exist_ok=True)

    if args.model_type == 'FFN':
        model_bleus = BLEU_evaluate_checkpoints_FFN(args.path_to_folder, args.beam_size, args.max_decoding_time_step, args.temperature)
    elif args.model_type == 'RNN':
        model_bleus = BLEU_evaluate_checkpoints_RNN(args.path_to_folder, args.beam_size, args.max_decoding_time_step, args.temperature)
    # Save the results
    if args.use_pickle:
        PickleLoader.save(args.dest_path, model_bleus)
    else:    
        with open(args.dest_path, 'w') as f:
            json.dump(model_bleus, f)
