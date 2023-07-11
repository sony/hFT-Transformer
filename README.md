# hFT-Transformer

This repository contains the official PyTorch implementation of **"Automatic Piano Transcription with Hierarchical Frequency-Time Transformer"** presented in ISMIR2023 ([arXiv 2307.04305](https://arxiv.org/abs/2307.04305)).

## Development Environment
- OS
  + Ubuntu 18.04
- memory
  + 32GB
- GPU
  + corpus generation, evaluation
    - NVIDIA GeForce RTX 2080 Ti
  + training
    - NVIDIA A100
- Python
  + 3.6.9
- Required Python libraries
  + [requirements.txt](requirements.txt)

## Usage
1) corpus generation (MAESTRO-V3)
```
$ ./corpus/EXE-CORPUS-MAESTRO.sh
```
2) training
```
$ ./training/EXE-TRAINING-MAESTRO.sh
```
3) evaluation

If you want to avoid training models from scratch, you can download and put the model under the `checkpoint/MAESTRO-V3` directory.

`model_016_003.pkl` is the model for MAESTRO.

```
$ wget https://github.com/sony/hFT-Transformer/releases/download/ismir2023/checkpoint.zip
$ unzip checkpoint.zip
$ ./evaluation/EXE-EVALUATION-MAESTRO.sh model_016_003.pkl test
```

If you want to evaluate the trained model using the validation set, you can change the second argument as below.
```
$ ./evaluation/EXE-EVALUATION-MAESTRO.sh model_016_003.pkl valid
```

## Citation
Keisuke Toyama, Taketo Akama, Yukara Ikemiya, Yuhta Takida, Wei-Hsiang Liao, and Yuki Mitsufuji, "Automatic Piano Transcription with Hierarchical Frequency-Time Transformer," in Proceedings of the 24th International Society for Music Information Retrieval Conference, 2023.
```
@inproceedings{toyama2023,
    author={Keisuke Toyama and Taketo Akama and Yukara Ikemiya and Yuhta Takida and Wei-Hsiang Liao and Yuki Mitsufuji},
    title={Automatic Piano Transcription with Hierarchical Frequency-Time Transformer},
    booktitle={Proceedings of the 24th International Society for Music Information Retrieval Conference},
    year={2023}
}
```

## Contact
- Keisuke Toyama (keisuke.toyama@sony.com)

## Reference
- PyTorch Seq2Seq (https://github.com/bentrevett/pytorch-seq2seq)
