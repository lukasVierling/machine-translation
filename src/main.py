import socket
HOSTNAME = socket.gethostname()

import argparse
import yaml
from train import train
from test import test

import os
import datetime
# Create argument parser
parser = argparse.ArgumentParser(description='Neural Network Training and Testing')

# Mode arguments
parser.add_argument('mode', choices=['train', 'test'], help='Mode: train or test')
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
parser.add_argument('--model_dir', type=str, help='Path to model directory')
parser.add_argument('--window_size', type=int, help='Window size')
parser.add_argument('--n_checkpoint', type=int, help='Number of epochs for checkpointing. Ignored, if checkpoints_per_epoch is 1 or larger')
parser.add_argument('--n_evaluate', type=int, help='Number of epochs for evaluation')
parser.add_argument('--checkpoints_per_epoch', type=int, help='Number of checkpoints per epoch, if multiple checkpoints should be created per epoch')

# Configuration and data directory arguments
parser.add_argument('--config', type=str, help='Path to configuration file', default='../config/config.yml')
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

# Parse the command-line arguments
args = parser.parse_args()

# Check if configuration file is provided
if args.config:
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

with open(args.log_dir + f'/{model_name}/hyperparameters.yml', 'w') as file:
    yaml.dump(vars(args), file)
    
with open(args.log_dir + f'/{model_name}/hyperparameters.yml', 'r') as file:
    config = yaml.safe_load(file)

# Set the appropriate function based on the mode
if args.mode == 'train':
    print('Start training...')
    func = train
else:
    print('Start testing...')
    func = test

# Call the function with the provided hyperparameters
func(config)
