#!/bin/bash

# Activate conda environment
source activate MT

# Navigate to the "src" folder
cd src

# Execute the command
python main.py train rnn --name 12k_bpe --device cuda --encoder_bidirectional --config ../config/rnn_config.yml --bpe_ops 12000
python evaluate.py --path_to_folder ../logs/12k_bpe/checkpoints --beam_size 20 --max_decoding_time_step 100 --temperature 1 --model_type RNN --dest_path ../results/optimization/12k_bpe