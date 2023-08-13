import socket
HOSTNAME = socket.gethostname()

import argparse
import yaml
from train import train
from rnn_train import rnn_train
from test import test

import os
import datetime
# Create argument parser
parser = argparse.ArgumentParser(description='Neural Network Training and Testing')

# Mode arguments
parser.add_argument('mode', choices=['train', 'test'], help='Mode: train or test')
parser.add_argument('model_type', choices=['rnn', 'ffn'], help='Model: rnn or ffn')
parser.add_argument('--name', type=str, help='Name of the model')
parser.add_argument('--device', type=str, help='Device to use for training/testing (cpu, cuda or mps)', default='cpu')

# Architecture arguments
parser.add_argument('--batch_size', type=int, help='Batch size')
parser.add_argument('--embedding_dim', type=int, help='Embedding dimension')
parser.add_argument('--linSource_dim', type=int, help='linSource dimension')
parser.add_argument('--linTarget_dim', type=int, help='linTarget dimension')
parser.add_argument('--lin1_dim', type=int, help='lin1 dimension')
parser.add_argument('--activation', choices=['gelu', 'relu', 'tanh', 'sigmoid'], help='Activation function')
parser.add_argument('--strategy', choices=['uniform', 'normal'], help='Initialization strategy')
parser.add_argument('--zero_bias', action='store_true', help='Flag indicating if bias should be initialized with zeros')

parser.add_argument('--encoder_embedding_size', type=int, help='Encoder embedding dimension')
parser.add_argument('--encoder_rnn_hidden_size', type=int, help='Encoder RNN hidden size')
parser.add_argument('--num_encoder_rnn_layers', type=int, help='Encoder number of RNN layers')
parser.add_argument('--encoder_dropout', type=float, help='Encoder dropout')
parser.add_argument('--encoder_bidirectional', action='store_true', help='Flag indicating if encoder RNN should be bidirectional')
parser.add_argument('--encoder_rnn_type', choices=['rnn', 'lstm', 'gru'], help='Encoder RNN type. Can be vanilla, lstm or gru')

parser.add_argument('--decoder_embedding_size', type=int, help='Decoder embedding dimension')
parser.add_argument('--decoder_rnn_hidden_size', type=int, help='Decoder RNN hidden size')
parser.add_argument('--num_decoder_rnn_layers', type=int, help='Decoder number of layers RNN')
parser.add_argument('--decoder_dropout', type=float, help='Decoder dropout')
parser.add_argument('--decoder_rnn_type', choices=['rnn', 'lstm', 'gru'], help='Decoder RNN type. Can be vanilla, lstm or gru')

parser.add_argument('--attention_function', choices=['dot_product', 'additive', 'none'], help='Attention function')
parser.add_argument('--attention_lin_size', type=int, help='Attention linear layer size for additive attention')
parser.add_argument('--teacher_forcing_ratio', type=float, help='Teacher forcing ratio')
parser.add_argument('--teacher_forcing_schedule', choices=['exponential_decay', 'linear_decay', 'constant'], help='Teacher forcing schedule method')

# Optimizer argument
parser.add_argument('--optimizer', choices=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta'], help='Optimizer')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--learning_rate_halving', action='store_true', help='Flag indicating if learning rate should be halved when dev set performance does not improve for k epochs')
parser.add_argument('--learning_rate_patience', type=int, help='Number of epochs to wait before halving the learning rate. Ignored, if learning_rate_halving is not set')
parser.add_argument('--learning_rate_performance_epsilon', type=float, help='Minimum improvement in dev set performance to halve the learning rate. Ignored, if learning_rate_halving is not set')
parser.add_argument('--momentum', type=float, help='Momentum for SGD optimizer. Ignored if optimizer is not SGD')
parser.add_argument('--weight_decay', type=float, help='Weight decay for SGD and RMSprop optimizers. Ignored if optimizer is not SGD or RMSprop')
parser.add_argument('--adam_beta1', type=float, help='Beta1 for Adam optimizer. Ignored if optimizer is not Adam')
parser.add_argument('--adam_beta2', type=float, help='Beta2 for Adam optimizer. Ignored if optimizer is not Adam')

# Training arguments
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--log_dir', type=str, help='Path to log directory')
parser.add_argument('--model_path', type=str, help='Path to model for training resumption')
parser.add_argument('--optimizer_path', type=str,help='Path to optimizer for training resumption')
parser.add_argument('--window_size', type=int, help='Window size')
parser.add_argument('--n_checkpoint', type=int, help='Number of epochs for checkpointing. Ignored, if checkpoints_per_epoch is 1 or larger')
parser.add_argument('--n_evaluate', type=int, help='Number of epochs for evaluation')
parser.add_argument('--checkpoints_per_epoch', type=int, help='Number of checkpoints per epoch, if multiple checkpoints should be created per epoch')
parser.add_argument('--resume_training', action='store_true', help='Flag indicating if training should be resumed from the given checkpoint')
parser.add_argument('--resume_epoch', type=int, help='Epoch to resume training from. Ignored, if resume_training is not set')

parser.add_argument('--clip', type=float, help='Gradient clipping value for RNN training')

# Configuration and data directory arguments
parser.add_argument('--config', type=str, help='Path to configuration file')
parser.add_argument('--train_source', type=str, help='Path to source data for training')
parser.add_argument('--train_target', type=str, help='Path to target data for training')
parser.add_argument('--dev_source', type=str, help='Path to source data for validation')
parser.add_argument('--dev_target', type=str, help='Path to target data for validation')
parser.add_argument('--test_source', type=str, help='Path to source data for testing')
parser.add_argument('--test_target', type=str, help='Path to target data for testing')

# BPE arguments
parser.add_argument('--bpe_ops', type=int, help='Number of BPE operations')
parser.add_argument('--joined_bpe', action='store_true', help='Flag indicating joined BPE')
parser.add_argument('--batch_alignment', type=str, help='Batch alignment strategy')

#BLEU and Search arguments
#parse beam size for search
parser.add_argument('--beam_size', type=int, help='Number of beams')
parser.add_argument('--max_decoding_time_step', type=int, help='Number of time steps for decoding during search')
parser.add_argument('--raw_dev_target', type=str, help='Path to the raw text data, unpreprocessed')

# Early stopping arguments
parser.add_argument('--early_stopping', action='store_true', help='Flag indicating if early stopping should be used')
parser.add_argument('--early_stopping_threshold', type=float, help='Threshold for early stopping. Ignored, if early_stopping is not set')

# Parse the command-line arguments
args = parser.parse_args()

# Check if configuration file is provided
if not args.config or not os.path.isfile(args.config):
    print('No configuration file provided. Using default values.')
    config_file = f"../config/{args.model_type}_config.yml"
    setattr(args, 'config', config_file)

# Load the configuration file
with open(args.config, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Update arguments with values from the configuration file
for key, value in config.items():
    if getattr(args, key, None) is None:
        setattr(args, key, value)

# Save the hyperparameters
if args.name is not None:
    model_name = args.name
else:
    model_name = f'model_{HOSTNAME}_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.name = model_name

# Create non-existing directories
os.makedirs(args.log_dir + f'/{model_name}', exist_ok=True)
os.makedirs(args.log_dir + f'/{model_name}/checkpoints', exist_ok=True)
os.makedirs(args.log_dir + f'/{model_name}/optimizer', exist_ok=True)

with open(args.log_dir + f'/{model_name}/hyperparameters.yml', 'w') as file:
    yaml.dump(vars(args), file)
    
with open(args.log_dir + f'/{model_name}/hyperparameters.yml', 'r') as file:
    config = yaml.safe_load(file)

# Set the appropriate function based on the mode
if args.mode == 'train':
    print('Start training...')
    if args.model_type == 'rnn':
        func = rnn_train
    elif args.model_type == 'ffn':
        func = train
else:
    print('Start testing...')
    func = test

# Call the function with the provided hyperparameters
func(config)
