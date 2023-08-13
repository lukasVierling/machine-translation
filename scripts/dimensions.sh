#!/bin/bash

# Define the arrays
embedding=(100 200)
lin1dim=(50 100 200)

# Iterate over the combinations
for value1 in "${embedding[@]}"; do
    for value2 in "${lin1dim[@]}"; do
        # Convert float value to string
        name=$(printf "%.4f" "$value1$value2")
        
        # Call the Python script with the name and learning_rate parameters
        python main.py train --config ../config/config.yml --name "$name" --epoch 6 --embedding_dim $value1 --lin1_dim $value2 --batch_size 64
    done
done