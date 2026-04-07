#!/bin/bash

python3 ../../tig.py generate_dataset vehicle_routing train.json --out tig_train
python3 ../../tig.py generate_dataset vehicle_routing test.json --out tig_test