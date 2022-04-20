#/!bin/bash

export PYTHONPATH=./:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3

seed=1029
dataset=cifar-100
num_classes=100
net1=vgg13
net2=MobileNetV2

python train_both.py \
    --config=configs/$dataset/seed-$seed/global-pkt/$net1-$net2.yml \
    --logdir=run/$dataset/seed-$seed/pkt_both_pretrained/$net1-$net2 \
    --file_name_cfg=kd-{}-{}-{}-{}-add-di
    --gpu_preserve False \
    --debug False
