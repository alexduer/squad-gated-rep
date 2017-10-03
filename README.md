# Replication of Gated Self-Matching Networks


## Content
This repository contains the code of a reimplementation of the 
"Gated self-matching networks for reading comprehension and question answering"
by Wang et al. which delivers strong performance on the SQuAD benchmark dataset.

The original paper:

```
@inproceedings{wang2017gated,
  title={Gated self-matching networks for reading comprehension and question answering},
  author={Wang, Wenhui and Yang, Nan and Wei, Furu and Chang, Baobao and Zhou, Ming},
  booktitle={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  volume={1},
  pages={189--198},
  year={2017}
}
```

A link to our replication paper will follow.


## Requirements
* Python 3.5
* Java 8


## Running the Replication
main.py offers a documented command line utility to our models.

```
python3 main.py --help
```

You can train the baseline model by running:

```
python3 main.py --model=RnetRep0
```

main_rep.py offers a script that reruns all experiments from our paper. You can use it by running:

```
python3 main_rep.py /path/to/save/output/to
```

The script will create a subfolder for each model which will contain a log file which you can use to monitor progress.

