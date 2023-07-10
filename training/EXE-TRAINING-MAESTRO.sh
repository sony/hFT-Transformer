#! /bin/bash

CURRENT_DIR=$(pwd)
FILE_CONFIG=${CURRENT_DIR}/corpus/MAESTRO-V3/dataset/config.json
DIR_DATASET=${CURRENT_DIR}/corpus/MAESTRO-V3/dataset

DIR_CHECKPOINT=${CURRENT_DIR}/checkpoint/MAESTRO-V3
mkdir -p $DIR_CHECKPOINT
python3 ${CURRENT_DIR}/training/m_training.py -config $FILE_CONFIG -d_out $DIR_CHECKPOINT -d_dataset $DIR_DATASET -n_div_train 4 -n_div_valid 1 -n_div_test 1 -epoch 20 -batch 8 -n_slice 16 -weight_A 1.0 -weight_B 1.0
