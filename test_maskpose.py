import torch
from torch import nn
import math
import torchvision
import torch.utils.model_zoo as model_zoo
# from torchsummary import summary
from models import MergeResNet, ResNet, FaceMask_ResNet
import torch.nn.functional as F
import utils
# import logging as logger
import shutil
import sys
from torchvision.models.resnet import BasicBlock
import cv2
import numpy as np
device = 'cuda:0'

def load_filtered_state_dict(model, state_dict):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

model = MergeResNet(BasicBlock, [2, 2, 2, 2])
model.to(device)
snapshot = "/snapshots/MaskPose_R18"
state_dict = torch.load(snapshot, map_location = device )
load_filtered_state_dict(model, state_dict)
model.eval()

idx_tensor = [idx for idx in range(66)]
idx_tensor = torch.Tensor(idx_tensor).to(device)

def get_information(img_path, model):
    image = cv2.imread(img_path)
    image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image.astype(np.float32)).to(device)
    mask, yaw, pitch, roll = model(image)
    yaw = utils.softmax_temperature(yaw.data, 1)
    yaw = torch.sum(yaw * idx_tensor, 1).to(device) * 3 - 99

    pitch = utils.softmax_temperature(pitch.data, 1)
    pitch = torch.sum(pitch * idx_tensor, 1).to(device) * 3 - 99

    roll = utils.softmax_temperature(roll.data, 1)
    roll = torch.sum(roll * idx_tensor, 1).to(device) * 3 - 99
    return torch.argmax(mask).item(), yaw, pitch, roll

img_path = ""
is_mask, yaw, pitch, roll = get_information(img_path, model)
print(is_mask, yaw, pitch, roll)