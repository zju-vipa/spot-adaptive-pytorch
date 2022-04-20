#/!bin/bash

export PYTHONPATH=./:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

seed=1029
dataset=imagenet
num_classes=1000
net1=vgg13
net2=MobileNetV2

python train_baseline.py \
    --config=configs/$dataset/seed-$seed/global-fitnet/$net1-$net2.yml \
    --logdir=run/$dataset/seed-$seed/baseline-fitnet-kd-1/$net1-$net2 \
    --file_name_cfg=kd-{}-{}-{}-{}
    --gpu_preserve False \
    --debug False

