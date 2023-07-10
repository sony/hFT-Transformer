#! /bin/bash

CURRENT_DIR=$(pwd)
FILE_CONFIG=${CURRENT_DIR}/corpus/MAPS/dataset/config.json
DIR_DATASET=${CURRENT_DIR}/corpus/MAPS/dataset

# 1-F-D-T (default)
DIR_CHECKPOINT_1FDT=${CURRENT_DIR}/checkpoint/MAPS/1FDT
mkdir -p $DIR_CHECKPOINT_1FDT
python3 ${CURRENT_DIR}/training/m_training.py -config $FILE_CONFIG -d_out $DIR_CHECKPOINT_1FDT -d_dataset $DIR_DATASET -n_div_train 1 -n_div_valid 1 -n_div_test 1 -epoch 50 -batch 8 -n_slice 16 -weight_A 1.0 -weight_B 1.0

# 2-F-D-T
DIR_CHECKPOINT_2FDT=${CURRENT_DIR}/checkpoint/MAPS/2FDT
mkdir -p $DIR_CHECKPOINT_2FDT
python3 ${CURRENT_DIR}/training/m_training_ablation.py -config $FILE_CONFIG -d_out $DIR_CHECKPOINT_2FDT -d_dataset $DIR_DATASET -n_div_train 1 -n_div_valid 1 -n_div_test 1 -epoch 50 -batch 8 -n_slice 16 -weight_A 1.0 -weight_B 1.0 -enc_alg CNNblock_SAfreq -dec_alg CAfreq_SAtime

# 1-F-D-N
DIR_CHECKPOINT_1FDN=${CURRENT_DIR}/checkpoint/MAPS/1FDN
mkdir -p $DIR_CHECKPOINT_1FDN
python3 ${CURRENT_DIR}/training/m_training_ablation.py -config $FILE_CONFIG -d_out $DIR_CHECKPOINT_1FDN -d_dataset $DIR_DATASET -n_div_train 1 -n_div_valid 1 -n_div_test 1 -epoch 50 -batch 8 -n_slice 16 -weight_A 1.0 -weight_B 1.0 -enc_alg CNNtime_SAfreq -dec_alg CAfreq

# 1-F-L-T
DIR_CHECKPOINT_1FLT=${CURRENT_DIR}/checkpoint/MAPS/1FLT
mkdir -p $DIR_CHECKPOINT_1FLT
python3 ${CURRENT_DIR}/training/m_training_ablation.py -config $FILE_CONFIG -d_out $DIR_CHECKPOINT_1FLT -d_dataset $DIR_DATASET -n_div_train 1 -n_div_valid 1 -n_div_test 1 -epoch 50 -batch 8 -n_slice 16 -weight_A 1.0 -weight_B 1.0 -enc_alg CNNtime_SAfreq -dec_alg linear_SAtime
