#/!bin/bash

export PYTHONPATH=./:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3

seed=1029
dataset=tiny-imagenet
num_classes=200
net1=resnet56
net2=resnet20

python test_policy.py \
    --config=configs/$dataset/seed-$seed/global-vid/$net1-$net2.yml \
    --gpu_preserve False \
    --debug False
