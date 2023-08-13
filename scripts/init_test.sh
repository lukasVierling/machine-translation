#!/bin/bash

# List of strategies
strategies=('uniform' 'normal')

# List of zero_bias values
zero_bias_values=(true false)

# Iterate over strategies and zero_bias values
for strategy in "${strategies[@]}"
do
  for zero_bias in "${zero_bias_values[@]}"
  do
    # Set the name parameter as the combination of strategy and zero_bias
    name="${strategy}_zero_bias_${zero_bias}"
    
    # Call the Python script with the name, strategy, and zero_bias parameters
    if [ "$zero_bias" = true ]; then
        python main.py train --name "$name" --strategy "$strategy" --zero_bias
    else
        python main.py train --name "$name" --strategy "$strategy"
    fi
  done
done
