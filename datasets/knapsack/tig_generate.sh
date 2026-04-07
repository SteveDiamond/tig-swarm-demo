#!/bin/bash

python3 ../../tig.py generate_dataset knapsack train.json --out tig_train
python3 ../../tig.py generate_dataset knapsack test.json --out tig_test