import os

import random

activations = ['gelu', 'tanh', 'relu']

optimizers = [{'name':'adam', 'lr': 0.001}, 
              {'name':'sgd', 'lr': 0.001},
              {'name':'adagrad', 'lr': 0.01}]

bpe_ops = [(1000, True), (7000, True), (7000, False), (15000, False)]


for i in range(1, 9):
    activation = random.choice(activations)
    optimizer = random.choice(optimizers)
    optimizer_name = optimizer['name']
    learning_rate = optimizer['lr']

    bpe_op, joined_bpe = random.choice(bpe_ops)

    source_train = f"../data/{bpe_op//1000}k_{'joined_' if joined_bpe else ''}BPE_indexed/train/source.train.pickle"
    target_train = f"../data/{bpe_op//1000}k_{'joined_' if joined_bpe else ''}BPE_indexed/train/target.train.pickle"
    source_dev = f"../data/{bpe_op//1000}k_{'joined_' if joined_bpe else ''}BPE_indexed/dev/source.dev.pickle"
    target_dev = f"../data/{bpe_op//1000}k_{'joined_' if joined_bpe else ''}BPE_indexed/dev/target.dev.pickle"

    cmd = f"python3 main.py train --name final_model_{i}a_{activation}_{optimizer_name}_{bpe_op}{'_joined' if joined_bpe else ''} --epochs 20 --learning_rate {learning_rate} --activation {activation} --optimizer {optimizer_name} --bpe_ops {bpe_op} {'--joined_bpe' if joined_bpe else ''} --train_source {source_train} --train_target {target_train} --dev_source {source_dev} --dev_target {target_dev}"
    cmd2 = f"python3 main.py train --name final_model_{i}b_{activation}_{optimizer_name}_{bpe_op}{'_joined' if joined_bpe else ''} --epochs 20 --learning_rate {learning_rate} --activation {activation} --optimizer {optimizer_name} --bpe_ops {bpe_op} {'--joined_bpe' if joined_bpe else ''} --train_source {source_train} --train_target {target_train} --dev_source {source_dev} --dev_target {target_dev}"


    print("Training model with ...")
    print(f"Activation: {activation}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Learning rate: {learning_rate}")
    print(f"BPE operations: {bpe_op}")
    print(f"Joined BPE: {joined_bpe}")
    print("")

    os.system(cmd)
    os.system(cmd2)