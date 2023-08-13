import sys
sys.path.append('..')

import argparse
import torch
import os

from src.pickle_loader import PickleLoader
from src.search.ffn_search import beam_search as ff_beam_search, greedy_search as ff_greedy_search
from src.search.rnn_search import beam_search as rnn_beam_search, greedy_search as rnn_greedy_search
from src.data_loader import DataLoader
from src.utils.byte_pair_encoding import BytePairEncoder
from src.preprocessing.dictionary import Dictionary

def main(parser):
    # parse arguments
    args = parser.parse_args()
    # load model

    model = torch.load(args.model_path, map_location=torch.device('cpu'))

    if args.model_type == 'rnn':
        encoder, decoder = model
        config = encoder.config
        greedy_search = rnn_greedy_search
        beam_search = rnn_beam_search
    else:
        config = model.config
        greedy_search = ff_greedy_search
        beam_search = ff_beam_search
    # load source text
    source_path = args.source_path
    if not source_path:
        source_path = "../data/raw/dev/source.dev"
    
    if not os.path.exists(source_path):
        raise ValueError("Source path does not exist")

    # load corpus
    source_loader = DataLoader(source_path)
    # load encoder
    source_encoder_path = f"../encoder/BPE_{'JOINED' if config['joined_bpe'] else 'DE'}_{config['bpe_ops']}.pickle"
    source_encoder = PickleLoader.load(source_encoder_path)
    target_encoder_path = f"../encoder/BPE_{'JOINED' if config['joined_bpe'] else 'EN'}_{config['bpe_ops']}.pickle"
    target_encoder = PickleLoader.load(target_encoder_path)

    #calc the ratio
    train_source_data = PickleLoader.load(f"../data/{config['bpe_ops']}_BPE_indexed/train/source.train.pickle")
    train_target_data = PickleLoader.load(f"../data/{config['bpe_ops']}_BPE_indexed/train/target.train.pickle")
    # get average sentence length for source and target
    source_average_sentence_len = sum([len(sentence) for sentence in train_source_data]) / len(train_source_data)
    target_average_sentence_len = sum([len(sentence) for sentence in train_target_data]) / len(train_target_data)
    source_target_ratio = source_average_sentence_len / target_average_sentence_len
    #print(source_target_ratio)
    # load dictionary
    source_dictionary_path = f"../dictionaries/dict_{'JOINED_' if config['joined_bpe'] else ''}DE_{config['bpe_ops']}.pickle"
    source_dictionary = PickleLoader.load(source_dictionary_path)

    target_dictionary_path = f"../dictionaries/dict_{'JOINED_' if config['joined_bpe'] else ''}EN_{config['bpe_ops']}.pickle"
    target_dictionary = PickleLoader.load(target_dictionary_path)

    # encode corpus
    source_encoded = source_encoder.encode_corpus(source_loader.load_data())
    #target_encoded = target_encoder.encode_corpus(loader.load_data())

    # indexed corpus
    source_indexed = source_dictionary.apply_mapping(source_encoded)
    #target_indexed = target_dictionary.apply_mapping(target_encoded)
    assert type(source_indexed) == list and type(source_indexed[0]) == list and type(source_indexed[0][0]) == int, "Source corpus is not indexed"

    n_best = min(args.n_best, args.beam_size) if args.n_best > 0 else args.beam_size

    # reference corpus if given
    if args.reference_path:
        reference_corpus = DataLoader(args.reference_path).tokenize(mode='lines')
    else:
        reference_corpus = [None for _ in range(len(source_indexed))]
    # loop over corpus
    for raw_line, indexed_line, reference_line in zip(source_loader.tokenize(mode='lines'), source_indexed, reference_corpus):
        # print source sentence
        raw_line_print = "" if args.translation_only else f"Source sentence: \n{raw_line}\n"
        if args.destination_path:
            with open(args.destination_path, 'a') as f:
                f.write(raw_line_print + "\n")
        else:
            print(raw_line_print)

        # print reference if given
        if reference_line and not args.translation_only:
            reference_print = f"Reference: \n{reference_line}\n"
            if args.destination_path:
                with open(args.destination_path, 'a') as f:
                    f.write(reference_print + "\n")
            else:
                print(reference_print)

        # translate
        if args.search_function == 'greedy':
            # greedy search
            translation = greedy_search(indexed_line, model, args.max_decoding_time_step)
            # remove mapping
            # wrap in list because greed search returns just a single list
            translation = target_dictionary.remove_mapping([translation])
            # remove bpe optionally
            if args.remove_bpe:
                translation = target_encoder.decode(translation)
            # print translation
            translation_print = f"Translation: \n{translation[0]}\n"
            if args.destination_path:
                with open(args.destination_path, 'a') as f:
                    f.write(translation_print + "\n")
            else:
                print(translation_print)
        elif args.search_function == 'beam':
            # beam search and choose n_best
            translations = beam_search(source=indexed_line, model=model, beam_size=args.beam_size, 
                                       max_decoding_time_step=args.max_decoding_time_step, 
                                       alignment_modeling=args.alignment_strategy, temperature=args.temperature, source_target_ratio=source_target_ratio)[:n_best]
            # remove mapping
            translations = target_dictionary.remove_mapping(translations)
            # remove bpe optionally
            if args.remove_bpe:
                translations = [target_encoder.decode(translation) for translation in translations]
            # print n best translations
            for i, sentence in enumerate(translations):
                #print(f"Translation {i+1}: \n{sentence}\n")
                translation_print = f"{sentence}" if args.translation_only else f"Translation {i+1}: \n{sentence}\n"
                if args.destination_path:
                    with open(f'{args.destination_path}', 'a') as f:
                        f.write(translation_print + "\n")
                else:
                    print(translation_print)
            
            # print line break
            if args.destination_path and not args.translation_only:
                with open(args.destination_path, 'a') as f:
                    f.write("\n-------------------\n\n")
            else:
                print("\n-------------------\n")
        else:
            raise ValueError("Invalid search function")


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(description="Score a corpus based on a given model, dictionary, encoder and alignment")
    # add arguments
    parser.add_argument("model_type", choices=["rnn", "ff"], help="Choose between RNN model_type and non-RNN model_type")

    parser.add_argument("model_path", type=str, help="Path to model")

    parser.add_argument("--source_path", type=str, help="Path to source corpus (raw)", default=None)

    parser.add_argument("--reference_path", type=str, help="Path to reference corpus (raw) (Optional for better comparision)", required=False)

    parser.add_argument('--search_function', choices=['greedy', 'beam'], default='beam',
                    help='Choose the search algorithm (default: greedy)')
    parser.add_argument('--beam_size', type=int, default=15, metavar='N',
                    help='Beam size for beam search (default: 5). Ignored if greedy search is chosen.')
    parser.add_argument('--max_decoding_time_step', type=int, default=100, metavar='N',
                    help='Maximum number of decoding time steps (default: 100).')
    parser.add_argument('--alignment_strategy', choices=['average'], default='average',
                        help='Choose the alignment strategy (default: average). Currently, only average is supported.')
    parser.add_argument("--n_best", type=int, default=0,
                        help="Number of best sentences to print. Should be smaller than beam size. If 0, print all sentences. If 1, print only the best sentence. Ignored if greedy search is chosen.")
    parser.add_argument('--remove_bpe', action='store_true', default=False,
                        help='Remove BPE from output (default: False)')
    parser.add_argument('--destination_path', type=str, default=None,
                        help='Path to destination file. If not provided, print to stdout.')
    # add argument for temperature for search
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling search (default: 1.0). Ignored if greedy search is chosen.')

    parser.add_argument('--translation_only', action='store_true', default=False, help='Print only the translations, i.e. for easier BLEU calculation')
    # run main
    main(parser)
