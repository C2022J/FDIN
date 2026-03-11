#!/usr/bin/env bash

CONFIG=$1
GPU=$2


torchrun --nproc_per_node=$GPU --master_port=4456 basicsr/train_stereo_sr.py -opt $CONFIG --launcher pytorch
