#! /bin/bash

## MAESTRO v3.0.0
CURRENT_DIR=$(pwd)

# 1. download MAESTRO v3.0.0 data and expand them
mkdir -p $CURRENT_DIR/corpus/MAESTRO-V3

FILE=./maestro-v3.0.0.zip
if test -f "$FILE"; then
    echo "$FILE exists, proceed to unzip"
else 
    echo "$FILE does not exist. Downloading..."    
    wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip ./
fi

unzip maestro-v3.0.0.zip -d $CURRENT_DIR/corpus/MAESTRO-V3
# $ ($CURRENT_DIR/corpus/MAESTRO-V3/maestro-v3.0.0)

# 2. make lists that include train/valid/test split
mkdir -p $CURRENT_DIR/corpus/MAESTRO-V3/list
python3 $CURRENT_DIR/corpus/make_list_maestro.py -i $CURRENT_DIR/corpus/MAESTRO-V3/maestro-v3.0.0/maestro-v3.0.0.csv -d_list $CURRENT_DIR/corpus/MAESTRO-V3/list

# 3. rename the files
mkdir -p $CURRENT_DIR/corpus/MAESTRO-V3/midi
mkdir -p $CURRENT_DIR/corpus/MAESTRO-V3/wav
python3 $CURRENT_DIR/corpus/rename_maestro.py -d_i $CURRENT_DIR/corpus/MAESTRO-V3/maestro-v3.0.0 -d_o $CURRENT_DIR/corpus/MAESTRO-V3 -d_list $CURRENT_DIR/corpus/MAESTRO-V3/list

# 4. convert wav to log-mel spectrogram
mkdir -p $CURRENT_DIR/corpus/MAESTRO-V3/feature
python3 $CURRENT_DIR/corpus/conv_wav2fe.py -d_list $CURRENT_DIR/corpus/MAESTRO-V3/list -d_wav $CURRENT_DIR/corpus/MAESTRO-V3/wav -d_feature $CURRENT_DIR/corpus/MAESTRO-V3/feature -config $CURRENT_DIR/corpus/config.json

# 5. convert midi to note
mkdir -p $CURRENT_DIR/corpus/MAESTRO-V3/note
python3 $CURRENT_DIR/corpus/conv_midi2note.py -d_list $CURRENT_DIR/corpus/MAESTRO-V3/list -d_midi $CURRENT_DIR/corpus/MAESTRO-V3/midi -d_note $CURRENT_DIR/corpus/MAESTRO-V3/note -config $CURRENT_DIR/corpus/config.json

# 6. convert note to label
mkdir -p $CURRENT_DIR/corpus/MAESTRO-V3/label
python3 $CURRENT_DIR/corpus/conv_note2label.py -d_list $CURRENT_DIR/corpus/MAESTRO-V3/list -d_note $CURRENT_DIR/corpus/MAESTRO-V3/note -d_label $CURRENT_DIR/corpus/MAESTRO-V3/label -config $CURRENT_DIR/corpus/config.json

# 7. convert txt to reference for evaluation
mkdir -p $CURRENT_DIR/corpus/MAESTRO-V3/reference
python3 $CURRENT_DIR/corpus/conv_note2ref.py -f_list $CURRENT_DIR/corpus/MAESTRO-V3/list/valid.list -d_note $CURRENT_DIR/corpus/MAESTRO-V3/note -d_ref $CURRENT_DIR/corpus/MAESTRO-V3/reference
python3 $CURRENT_DIR/corpus/conv_note2ref.py -f_list $CURRENT_DIR/corpus/MAESTRO-V3/list/test.list -d_note $CURRENT_DIR/corpus/MAESTRO-V3/note -d_ref $CURRENT_DIR/corpus/MAESTRO-V3/reference

# 8. make dataset
mkdir -p $CURRENT_DIR/corpus/MAESTRO-V3/dataset
python3 $CURRENT_DIR/corpus/make_dataset.py -f_config_in $CURRENT_DIR/corpus/config.json -f_config_out $CURRENT_DIR/corpus/MAESTRO-V3/dataset/config.json -d_dataset $CURRENT_DIR/corpus/MAESTRO-V3/dataset -d_list $CURRENT_DIR/corpus/MAESTRO-V3/list -d_feature $CURRENT_DIR/corpus/MAESTRO-V3/feature -d_label $CURRENT_DIR/corpus/MAESTRO-V3/label -n_div_train 4 -n_div_valid 1 -n_div_test 1
