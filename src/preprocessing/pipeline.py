import sys
sys.path.append("..")
#Import dataloader, bpe and dictionary
from src.data_loader import DataLoader
from src.utils.byte_pair_encoding import BytePairEncoder
from src.preprocessing.dictionary import Dictionary

import argparse
import os

from src.pickle_loader import PickleLoader

#parse arguments: bpe operations, joined bpe, source and target path
parser = argparse.ArgumentParser()
parser.add_argument("--bpe_ops", type=int, default=7000)
parser.add_argument("--joined_bpe", type=bool, default=False)
parser.add_argument("--source_path_train", type=str, default="../data/raw/train/source.train")
parser.add_argument("--target_path_train", type=str, default="../data/raw/train/target.train")
parser.add_argument("--source_path_dev", type=str, default="../data/raw/dev/source.dev")
parser.add_argument("--target_path_dev", type=str, default="../data/raw/dev/target.dev")
args = parser.parse_args()
# Load the data for train and dev
print(f"Joined BPE: {args.joined_bpe}")

source_train_data_loader = DataLoader(args.source_path_train)
target_train_data_loader = DataLoader(args.target_path_train)
source_dev_data_loader = DataLoader(args.source_path_dev)
target_dev_data_loader = DataLoader(args.target_path_dev)

# Fit BPE
if args.joined_bpe:
    merged_data = source_train_data_loader.load_data() + "\n" + target_train_data_loader.load_data()
    joined_bpe = BytePairEncoder()
    joined_bpe.fit(merged_data, operations=args.bpe_ops)
    source_bpe = joined_bpe
    target_bpe = joined_bpe
else:
    source_bpe = BytePairEncoder()
    source_bpe.fit(source_train_data_loader.load_data(), operations=args.bpe_ops)
    target_bpe = BytePairEncoder()
    target_bpe.fit(target_train_data_loader.load_data(), operations=args.bpe_ops)

# Encode corpus
source_train_data_encoded = source_bpe.encode_corpus(source_train_data_loader.load_data())
target_train_data_encoded = target_bpe.encode_corpus(target_train_data_loader.load_data())
source_dev_data_encoded = source_bpe.encode_corpus(source_dev_data_loader.load_data())
target_dev_data_encoded = target_bpe.encode_corpus(target_dev_data_loader.load_data())

# Dictionary
source_dictionary = Dictionary()
target_dictionary = Dictionary()
print(source_train_data_encoded)
source_dictionary.update(source_train_data_encoded)
print(source_dictionary.get_vocab_size())
target_dictionary.update(target_train_data_encoded)
source_train_data_indexed = source_dictionary.apply_mapping(source_train_data_encoded)
target_train_data_indexed = target_dictionary.apply_mapping(target_train_data_encoded)
source_dev_data_indexed = source_dictionary.apply_mapping(source_dev_data_encoded)
target_dev_data_indexed = target_dictionary.apply_mapping(target_dev_data_encoded)

# Store everything
if args.joined_bpe:
    if not os.path.exists(f"../data/{args.bpe_ops}_joined_BPE_indexed"):
        os.makedirs(f"../data/{args.bpe_ops}_joined_BPE_indexed")
    if not os.path.exists(f"../data/{args.bpe_ops}_joined_BPE_indexed/dev"):
        os.makedirs(f"../data/{args.bpe_ops}_joined_BPE_indexed/dev")
    if not os.path.exists(f"../data/{args.bpe_ops}_joined_BPE_indexed/train"):
        os.makedirs(f"../data/{args.bpe_ops}_joined_BPE_indexed/train")


    PickleLoader.save(f"../encoder/BPE_JOINED_{args.bpe_ops}.pickle", joined_bpe)
    PickleLoader.save(f"../dictionaries/dict_JOINED_DE_{args.bpe_ops}.pickle", source_dictionary)
    PickleLoader.save(f"../dictionaries/dict_JOINED_EN_{args.bpe_ops}.pickle", target_dictionary)
    PickleLoader.save(f"../data/{args.bpe_ops}_joined_BPE_indexed/train/source.train.pickle", source_train_data_indexed)
    PickleLoader.save(f"../data/{args.bpe_ops}_joined_BPE_indexed/train/target.train.pickle", target_train_data_indexed)
    PickleLoader.save(f"../data/{args.bpe_ops}_joined_BPE_indexed/dev/source.dev.pickle", source_dev_data_indexed)
    PickleLoader.save(f"../data/{args.bpe_ops}_joined_BPE_indexed/dev/target.dev.pickle", target_dev_data_indexed)

else:
    if not os.path.exists(f"../data/{args.bpe_ops}_BPE_indexed"):
        os.makedirs(f"../data/{args.bpe_ops}_BPE_indexed")
    if not os.path.exists(f"../data/{args.bpe_ops}_BPE_indexed/dev"):
        os.makedirs(f"../data/{args.bpe_ops}_BPE_indexed/dev")
    if not os.path.exists(f"../data/{args.bpe_ops}_BPE_indexed/train"):
        os.makedirs(f"../data/{args.bpe_ops}_BPE_indexed/train")

    PickleLoader.save(f"../encoder/BPE_DE_{args.bpe_ops}.pickle", source_bpe)
    PickleLoader.save(f"../encoder/BPE_EN_{args.bpe_ops}.pickle", target_bpe)
    PickleLoader.save(f"../dictionaries/dict_DE_{args.bpe_ops}.pickle", source_dictionary)
    PickleLoader.save(f"../dictionaries/dict_EN_{args.bpe_ops}.pickle", target_dictionary)

    PickleLoader.save(f"../data/{args.bpe_ops}_BPE_indexed/train/source.train.pickle", source_train_data_indexed)
    PickleLoader.save(f"../data/{args.bpe_ops}_BPE_indexed/train/target.train.pickle", target_train_data_indexed)
    PickleLoader.save(f"../data/{args.bpe_ops}_BPE_indexed/dev/source.dev.pickle", source_dev_data_indexed)
    PickleLoader.save(f"../data/{args.bpe_ops}_BPE_indexed/dev/target.dev.pickle", target_dev_data_indexed)
