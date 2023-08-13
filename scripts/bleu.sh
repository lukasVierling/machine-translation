#!/bin/bash

# Activate conda environment
source activate MT

# Navigate to the "src" folder
cd src

# Execute the command
python evaluate.py --path_to_folder ../logs/dropout/checkpoints --beam_size 20 --max_decoding_time_step 100 --temperature 1 --model_type RNN --dest_path ../results/optimization/dropout
python evaluate.py --path_to_folder ../logs/400_embed/checkpoints --beam_size 20 --max_decoding_time_step 100 --temperature 1 --model_type RNN --dest_path ../results/optimization/400_embed