#!/usr/bin/env bash

CONFIG=$1
GPU=$2


torchrun --nproc_per_node=$GPU --master_port=3780 basicsr/train.py -opt $CONFIG --launcher pytorch
