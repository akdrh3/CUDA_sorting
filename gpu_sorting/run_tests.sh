#!/bin/bash
# Save this as runAllTests.sh

# Array of numbers
numbers=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)

# Iterate over each number in the array
for number in "${numbers[@]}"
do
    # Run numberGenerate.py with the current number
    echo "Running numberGenerate.py with $number"
    python3 numberGenerate.py $number

    # Run testQuickSort with numbers.txt and capture output
    output=$(echo "numbers.txt" | ./testQuickSort)

    # Print and save output
    echo "$output"
    echo "$output" >> output.txt

    # Separate outputs with a line
    echo "--------------------------------------"
    echo "--------------------------------------" >> output.txt
done