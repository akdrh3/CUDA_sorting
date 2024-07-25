#!/bin/bash

# Initial dataset size
dataset_size=1

# Maximum dataset size
max_dataset_size=8192

# Output file for results
output_file="sorting_results.txt"

# Append header if the file is empty
if [ ! -s $output_file ]; then
    echo "Dataset Size, Number of Integers, Time Taken (ms)" > $output_file
fi

while [ $dataset_size -le $max_dataset_size ]
do
    # Generate the dataset
    echo "Generating dataset of size $dataset_size..."
    echo "$dataset_size" | python3 numberGenerate.py

    # Run the sorting test
    echo "Running ./testQuickSort for dataset of size $dataset_size..."
    result=$(echo "numbers.txt" | ./testQuickSort)
    
    # Extract number of integers and time taken
    num_integers=$(echo "$result" | grep "Number of integers in the file" | awk '{print $7}')
    time_taken=$(echo "$result" | grep "Time elipsed to copy array to gpu" | awk '{print $7}')
    
    # Print the results
    echo "Dataset size: $dataset_size, Number of integers: $num_integers, Time taken: $time_taken ms"

    # Save the results
    echo "$dataset_size, $num_integers, $time_taken" >> $output_file

    # Double the dataset size for the next iteration
    dataset_size=$((dataset_size * 2))
done

echo "Test completed. Results are saved in $output_file."
