#/!bin/bash

export PYTHONPATH=./:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

seed=1029
dataset=tiny-imagenet
num_classes=200
net1=wrn_40_2
net2=ShuffleNetV1

python train_reverse.py \
    --config=configs/$dataset/seed-$seed/global-nst/$net1-$net2.yml \
    --logdir=run/$dataset/seed-$seed/reverse_nst_kd/$net1-$net2 \
    --file_name_cfg=kd-{}-{}-{}-{}-add-di
    --gpu_preserve False \
    --debug False

