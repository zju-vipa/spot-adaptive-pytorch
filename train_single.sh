#/!bin/bash

export PYTHONPATH=./:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

seed=1029
dataset=imagenet
num_classes=1000
model=resnet56



python train_single.py \
    --config=configs/$dataset/seed-$seed/single/$model.yml \
    --logdir=run/$dataset/seed-$seed/single/$model \
    --file_name_cfg=kd-{}-{}
