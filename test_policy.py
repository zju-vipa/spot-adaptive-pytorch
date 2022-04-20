import os
import argparse
from typing import Dict, Any
import copy
import logging

import yaml

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module, ModuleDict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from dataset import get_dataset
from models import get_model
from distiller_zoo import get_loss_module, get_loss_forward
from optim import get_optimizer
from models.classifier import LinearClassifier
from models.gs import gumbel_softmax

from helper.util import str2bool, get_logger, preserve_memory, adjust_learning_rate_stage
from helper.util import make_deterministic
from helper.util import AverageMeter, accuracy
from helper.validate import validate_policy, validate


def get_dataloader(cfg: Dict[str, Any]):
    # dataset
    dataset_cfg = cfg["dataset"]
    train_dataset = get_dataset(split="train", **dataset_cfg)
    val_dataset = get_dataset(split="val", **dataset_cfg)
    num_classes = len(train_dataset.classes)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        num_workers=cfg["training"]["num_workers"],
        shuffle=False,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=cfg["validation"]["num_workers"],
        shuffle=False,
        pin_memory=True
    )
    return train_loader, val_loader, num_classes, train_dataset, val_dataset


def get_teacher(cfg: Dict[str, Any], num_classes: int) -> Module:
    teacher_cfg = copy.deepcopy(cfg["kd"]["teacher"])
    teacher_name = teacher_cfg["name"]
    ckpt_fp = teacher_cfg["checkpoint"]
    teacher_cfg.pop("name")
    teacher_cfg.pop("checkpoint")

    # load state dict
    state_dict = torch.load(ckpt_fp, map_location="cpu")["model"]


    model_t = get_model(
        model_name=teacher_name,
        num_classes=num_classes,
        state_dict=state_dict,
        **teacher_cfg
    )
    return model_t


def get_student(cfg: Dict[str, Any], num_classes: int) -> Module:
    student_cfg = copy.deepcopy(cfg["kd"]["student"])
    student_name = student_cfg["name"]
    student_cfg.pop("name")

    state_dict = None
    if "checkpoint" in student_cfg.keys():
        state_dict = torch.load(student_cfg["checkpoint"], map_location="cpu")["model"]
        student_cfg.pop("checkpoint")

    model_s = get_model(
        model_name=student_name,
        num_classes=num_classes,
        state_dict=state_dict,
        **student_cfg
    )
    return model_s


def test_kd(
        cfg: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        module_dict: ModuleDict,
        device: torch.device,
        dataset: Dataset
):
    model_t = module_dict["teacher"].eval()
    model_s = module_dict["student"].eval()
    policy = module_dict["policy"].eval()

    res = []

    for idx, (x, target) in enumerate(train_loader):
        x = x.to(device)
        target = target.to(device)

        with torch.no_grad():
            feat_t, logit_t = model_t(x, begin=0, end=100, is_feat=True)
            feat_s, logit_s = model_s(x, begin=0, end=100, is_feat=True)

            policy_feat = torch.cat((feat_t[-1].clone().detach(), feat_s[-1].clone().detach()), 1)
            policy_res = policy(policy_feat)

            action = gumbel_softmax(policy_res.view(policy_res.size(0), -1, 2))

            ac_middle = [action[:, i, 0].contiguous().float().view(-1).cpu().numpy() for i in range(action.shape[1])]

            # print(ac_middle)
            if ac_middle[0][0]<0.5:
                res.append(idx)
                print(dataset.get_path(idx))

    # val_acc, val_acc_top5, val_loss=validate(
    #     val_loader=val_loader,
    #     model=module_dict["student"],
    #     criterion=nn.CrossEntropyLoss().to(device),
    #     device=device
    # )
    # print(val_acc, val_acc_top5)

    print(res)
    return res

def main(
        cfg_filepath: str,
        gpu_preserve: bool = False,
        debug: bool = False
):
    with open(cfg_filepath) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    seed = cfg["training"]["seed"]

    # set seed
    make_deterministic(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get dataloaders
    train_loader, val_loader, num_classes, train_dataset, val_dataset = get_dataloader(cfg)

    # get models
    model_t = get_teacher(cfg, num_classes).to(device)
    model_s = get_student(cfg, num_classes).to(device)
    policy = LinearClassifier(model_t.get_hint_channel()+model_s.get_hint_channel(), 8).to(device)

    path = './run/tiny-imagenet/seed-1029/multi_vid_01_sss/resnet56-resnet20/ckpt/best.pth'

    state_dict = torch.load(path, map_location="cpu")["policy"]
    policy.load_state_dict(state_dict)

    state_dict = torch.load(path, map_location="cpu")["model"]
    model_s.load_state_dict(state_dict)

    model_t.eval()
    model_s.eval()
    policy.eval()

    module_dict = nn.ModuleDict(dict(
        teacher=model_t,
        student=model_s,
        policy=policy
    ))

    res_target = test_kd(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        module_dict=module_dict,
        device=device,
        dataset=train_dataset
    )

    res_path = [train_dataset.get_path(each) for each in res_target]
    print(res_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--gpu_preserve", type=str2bool, default=False)
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--preserve_percent", type=float, default=0.95)
    args = parser.parse_args()

    __global_values__ = dict(it=0)
    main(
        cfg_filepath=args.config,
        gpu_preserve=args.gpu_preserve,
        debug=args.debug
    )
