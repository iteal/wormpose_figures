#!/bin/bash

git clone https://github.com/iteal/wormpose_data.git

python3 wormpose_data/datasets/tierpsy/download_from_zenodo.py wormpose_data/datasets/tierpsy/N2.csv
python3 wormpose_data/datasets/tierpsy/download_from_zenodo.py wormpose_data/datasets/tierpsy/AQ2934.csv

ls N2
ls AQ2934