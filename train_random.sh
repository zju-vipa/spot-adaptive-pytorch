#/!bin/bash

export PYTHONPATH=./:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1

seed=1029
dataset=cifar-100
num_classes=100
net1=wrn_40_2
net2=ShuffleNetV1

python train_random.py \
    --config=configs/$dataset/seed-$seed/global-sp/$net1-$net2.yml \
    --logdir=run/$dataset/seed-$seed/random_sp/$net1-$net2 \
    --file_name_cfg=kd-{}-{}-{}-{}-add-di
    --gpu_preserve False \
    --debug False
#/!bin/bash
