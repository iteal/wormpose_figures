#!/bin/bash -l
#SBATCH -t 72:00:00
#SBATCH -c 36
#SBATCH --output=out/out-%j.out
#SBATCH --error=out/err-%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:V100:1

module load python/3.7.3

TEMP_DIR=/scratch

WORK_DIR=$(mktemp -d -p $TEMP_DIR)

wget https://raw.githubusercontent.com/iteal/wormpose/master/requirements.txt -O $WORK_DIR/requirements.txt
wget https://raw.githubusercontent.com/iteal/wormpose/master/extras/EigenWorms.csv -O $WORK_DIR/EigenWorms.txt

python3 -m venv $WORK_DIR/venv --clear
source $WORK_DIR/venv/bin/activate 

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools
python3 -m pip install -r $WORK_DIR/requirements.txt
python3 -m pip install wormpose

wormpose -v

RANDOM_SEED=1234

PYTHONHASHSEED=0 wormpose datagen tierpsy "$1" --temp_dir $TEMP_DIR --image_size 128 --random_seed $RANDOM_SEED
PYTHONHASHSEED=0 wormpose train "$1" --random_seed $((RANDOM_SEED + 1))
PYTHONHASHSEED=0 wormpose predict "$1" --temp_dir $TEMP_DIR --random_seed $((RANDOM_SEED + 2))
PYTHONHASHSEED=0 wormpose postprocess "$1"  --eigenworms_matrix_path $WORK_DIR/EigenWorms.txt
PYTHONHASHSEED=0 wormpose export "$1"
PYTHONHASHSEED=0 wormpose evaluate "$1" --temp_dir $TEMP_DIR --num_samples 600000 --random_seed $((RANDOM_SEED + 3))  --eigenworms_matrix_path $WORK_DIR/EigenWorms.txt
PYTHONHASHSEED=0 wormpose calibrate tierpsy "$1" --image_size 128 --num_samples 600000 --random_seed $((RANDOM_SEED + 4))
PYTHONHASHSEED=0 wormpose viz "$1" --temp_dir $TEMP_DIR


rm -rf $WORK_DIR