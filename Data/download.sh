#!/usr/bin/env bash

# Download SQuAD
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json 

# Download GloVe
wget http://nlp.stanford.edu/data/glove.6B.zip
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.6B.zip
unzip glove.840B.300d.zip

