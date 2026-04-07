#!/bin/bash

python3 ../../tig.py generate_dataset job_scheduling train.json --out tig_train
python3 ../../tig.py generate_dataset job_scheduling test.json --out tig_test