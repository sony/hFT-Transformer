#! /bin/bash

## MAPS
CURRENT_DIR=$(pwd)

# 1. download MAPS data and expand them
mkdir -p $CURRENT_DIR/corpus/MAPS
# (download MAPS dataset by yourself, then expand them to $CURRENT_DIR/corpus/MAPS/original)

# 2. rename the files
mkdir -p $CURRENT_DIR/corpus/MAPS/midi
mkdir -p $CURRENT_DIR/corpus/MAPS/wav
python3 $CURRENT_DIR/corpus/rename_maps.py -d_i $CURRENT_DIR/corpus/MAPS/original -d_o $CURRENT_DIR/corpus/MAPS -o $CURRENT_DIR/corpus/MAPS/MAPS_number.tsv

# 3. make lists that include train/valid/test split
mkdir -p $CURRENT_DIR/corpus/MAPS/list
python3 $CURRENT_DIR/corpus/make_list_maps.py -f_number $CURRENT_DIR/corpus/MAPS/MAPS_number.tsv -d_list $CURRENT_DIR/corpus/MAPS/list -data MUS

# 4. convert wav to log-mel spectrogram
mkdir -p $CURRENT_DIR/corpus/MAPS/feature
python3 $CURRENT_DIR/corpus/conv_wav2fe.py -d_list $CURRENT_DIR/corpus/MAPS/list -d_wav $CURRENT_DIR/corpus/MAPS/wav -d_feature $CURRENT_DIR/corpus/MAPS/feature -config $CURRENT_DIR/corpus/config.json

# 5. convert midi to note
mkdir -p $CURRENT_DIR/corpus/MAPS/note
python3 $CURRENT_DIR/corpus/conv_midi2note.py -d_list $CURRENT_DIR/corpus/MAPS/list -d_midi $CURRENT_DIR/corpus/MAPS/midi -d_note $CURRENT_DIR/corpus/MAPS/note -config $CURRENT_DIR/corpus/config.json

# 6. convert note to label
mkdir -p $CURRENT_DIR/corpus/MAPS/label
python3 $CURRENT_DIR/corpus/conv_note2label.py -d_list $CURRENT_DIR/corpus/MAPS/list -d_note $CURRENT_DIR/corpus/MAPS/note -d_label $CURRENT_DIR/corpus/MAPS/label -config $CURRENT_DIR/corpus/config.json

# 7. convert txt to reference for evaluation
mkdir -p $CURRENT_DIR/corpus/MAPS/reference
python3 $CURRENT_DIR/corpus/conv_note2ref.py -f_list $CURRENT_DIR/corpus/MAPS/list/valid.list -d_note $CURRENT_DIR/corpus/MAPS/note -d_ref $CURRENT_DIR/corpus/MAPS/reference
python3 $CURRENT_DIR/corpus/conv_note2ref.py -f_list $CURRENT_DIR/corpus/MAPS/list/test.list -d_note $CURRENT_DIR/corpus/MAPS/note -d_ref $CURRENT_DIR/corpus/MAPS/reference

# 8. make dataset
mkdir -p $CURRENT_DIR/corpus/MAPS/dataset
python3 $CURRENT_DIR/corpus/make_dataset.py -f_config_in $CURRENT_DIR/corpus/config.json -f_config_out $CURRENT_DIR/corpus/MAPS/dataset/config.json -d_dataset $CURRENT_DIR/corpus/MAPS/dataset -d_list $CURRENT_DIR/corpus/MAPS/list -d_feature $CURRENT_DIR/corpus/MAPS/feature -d_label $CURRENT_DIR/corpus/MAPS/label -n_div_train 1 -n_div_valid 1 -n_div_test 1
