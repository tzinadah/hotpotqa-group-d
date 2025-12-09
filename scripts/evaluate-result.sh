#!/bin/sh

# Takes the results name and evaluates it

# Extract the filepath from its name
FILEPATH="./predictions/$1.json"

echo evaluating $FILEPATH

# Output filepath
OUTPUT_PATH="./results/$1-results.json"

# Run the script and extract last line to the results to ignore error messages
python ./src/hotpotqa_group_d/evaluation/evaluation.py $FILEPATH ./data/hotpot_dev_fullwiki_v1.json | tail -n 1 >$OUTPUT_PATH

echo Saved results of $FILEPATH to $OUTPUT_PATH
