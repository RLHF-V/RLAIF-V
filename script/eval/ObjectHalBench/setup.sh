#!/bin/bash

#Download generated sentences
wget https://people.eecs.berkeley.edu/~lisa_anne/hallucination/generated_sentences.zip
unzip generated_sentences.zip
rm -r generated_sentences.zip

#Download
mkdir output
mkdir output/hallucination
wget https://people.eecs.berkeley.edu/~lisa_anne/hallucination/intermediate_image.zip
unzip intermediate_image.zip
rm -r intermediate_image.zip

cd data
wget https://people.eecs.berkeley.edu/~lisa_anne/hallucination/gt_labels.p
wget https://people.eecs.berkeley.edu/~lisa_anne/hallucination/vocab.p
wget https://people.eecs.berkeley.edu/~lisa_anne/hallucination/human_scores.zip
unzip human_scores.zip
rm -r human_scores.zip

cd ..