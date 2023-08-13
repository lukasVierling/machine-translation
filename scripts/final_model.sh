#!/bin/bash

# Iterate over the float values
for value in 
do
  # Convert float value to string
  name=$(printf "%.4f" "$value")
  
  # Call the Python script with the name and learning_rate parameters
  python main.py train --config ../config/config.yml --name "final_model_$name" --epoch 1 --batch_size 1000
done