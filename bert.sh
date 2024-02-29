#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

dataset=fewcomm 
dataset_mode=IO
N=5
K=1
python -u main.py --multi_margin --use_proto_as_neg --model bert --dataset $dataset --dataset_mode $dataset_mode  --trainN $N --N $N --K $K --Q 1 --trainable_margin_init 6.5
