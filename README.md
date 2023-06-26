# hFT-Transformer

AMT code with PyTorch library for ISMIR2023

## Title and Authors
"Automatic Piano Transcription with Hierarchical Frequency-Time Transformer"

_Authors: Keisuke Toayama, Taketo Akama, Yukara Ikemiya, Yuhta Takida, Wei-Hsiang Liao, and Yuki Mitsufuji_

## Development Environment
- OS
  + Ubuntu 18.04
- Python
  + 3.6.9
- Required Python libraries
  + numpy
  + torch
  + torchaudio 
  + mido
  + pretty_midi
  + mir_eval

## Usage
1) generate corpus (MAESTRO-V3)
```
$ cd corpus
$ ./EXE-CORPUS-MAESTRO.sh
```
2) training
```
$ cd ../training
$ ./EXE-TRAINING-MAESTRO.sh
```
3) evaluation
```
$ cd ../evaluation
$ ./EXE-EVALUATION-MAESTRO.sh
```

## Contact
- Keisuke Toyama (keisuke.toyama@sony.com)

## Reference
- PyTorch Seq2Seq (https://github.com/bentrevett/pytorch-seq2seq)
