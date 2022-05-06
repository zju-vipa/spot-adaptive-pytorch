# Spot-adaptive Knowledge Distillation
## Introduction
Knowledge distillation (KD) has become a well established paradigm for compressing deep neural networks. The typical way of conducting knowledge distillation is to train the student network under the supervision of the teacher network to harness the knowledge at one or multiple spots (i.e., layers) in the teacher network. The distillation spots, once specified, will not change for all the training samples, throughout the whole distillation process. In this work, we argue that distillation spots should be adaptive to training samples and distillation epochs. We thus propose a new distillation strategy, termed spot-adaptive KD (SAKD), to adaptively determine the distillation spots in the teacher network per sample, at every training iteration during the whole distillation period. As SAKD actually focuses on "where to distill" instead of "what to distill" that is widely investigated by most existing works, it can be seamlessly integrated into existing distillation methods to further improve their performance. Extensive experiments with 10 state-of-the-art distillers are conducted to demonstrate the effectiveness of SAKD for improving their distillation performance, under both homogeneous and heterogeneous distillation settings.

The work is accepted by IEEE Transactions on Image Processing, 2022.

```bibtex
  @ARTICLE{9767610,
        author={Song, Jie and Chen, Ying and Ye, Jingwen and Song, Mingli},
        journal={IEEE Transactions on Image Processing}, 
        title={Spot-adaptive Knowledge Distillation}, 
        year={2022},
        doi={10.1109/TIP.2022.3170728}
    }
```

This repo contains the code of the work. We benchmark 11 state-of-the-art knowledge distillation methods with spot-adaptive KD in PyTorch, including: 

- (FitNet) - Fitnets: hints for thin deep nets
- (AT) - Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
- (SP) - Similarity-Preserving Knowledge Distillation
- (CC) - Correlation Congruence for Knowledge Distillation
- (VID) - Variational Information Distillation for Knowledge Transfer
- (RKD) - Relational Knowledge Distillation
- (PKT) - Probabilistic Knowledge Transfer for deep representation learning
- (FT) - Paraphrasing Complex Network: Network Compression via Factor Transfer
- (FSP) - A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
- (NST) - Like what you like: knowledge distill via neuron selectivity transfer
- (CRD) - Contrastive Representation Distillation

## Installation
This repo was tested with Ubuntu 16.04.6 LTS, Python 3.6. 
And it should be runnable with PyTorch versions >= 0.4.0.

## Running
1.Fetch the pretrained teacher models by: 
```
sh train_single.sh 
```
which will run the code and save the models to <code> ./run/$dataset/$seed/$model/ckpt </code>

The flags in <code>train_single.sh</code> can be explained as:
- <code>seed</code>: specify the random seed.
- <code>dataset</code>: specify the training dataset.
- <code>num_classes</code>: give the number of categories of the above dataset.
- <code>model</code>: specify the model, see <code>'models/__init__.py'</code> to check the available model types.

Note: the default setting can be seen in config files from <code>'configs/$dataset/seed-$seed/single/$model.yml'</code>. 



2.Run our spot-adaptive KD by:
```
sh train.sh
```


3.(Optional) run the anti spot-adaptive KD by:
```
sh train_anti.sh
```

The flags in <code>train.sh</code> and <code>train_anti.sh</code> can be explained as:
- <code>seed</code>: specify the random seed.
- <code>dataset</code>: specify the training dataset.
- <code>num_classes</code>: give the number of categories of the above dataset.
- <code>net1</code>: specify the teacher model, see <code>'models/__init__.py'</code> to check the available model types.
- <code>net2</code>: specify the student model, see <code>'models/__init__.py'</code> to check the available model types.

Note: the default setting can be seen in config files from <code>'configs/$dataset/seed-$seed/$distiller/$net1-$net2.yml'</code>. 



