#!/bin/bash
# Save this as runAllTests.sh

# Array of numbers
numbers=(1 2 4)

# Iterate over each number in the array
for number in "${numbers[@]}"
do
    # Run numberGenerate.py with the current number
    echo "Running numberGenerate.py with $number"
    echo $number | python3 numberGenerate.py

    # Run mergesort with numbers.txt and capture output
    output=$(echo "numbers.txt" | ./mergesort 2>> merge_error_log.txt)

    # Print and save output
    echo "$output"
    echo "$output" >> mergeoutput.txt

    # Separate outputs with a line
    echo "--------------------------------------"
    echo "--------------------------------------" >> mergeoutput.txt
done