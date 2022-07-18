from plistlib import load
from random import shuffle
import torch
from torch import nn
from torch.utils.data import dataset, random_split, DataLoader
from datasets import FaceMaskDataset
from models import ResNet, BoTNet50_2, Res2Net, Bottle2neck
from torchvision.models.resnet import BasicBlock, Bottleneck
import sys
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import os
from torchsummary import summary
from utils import AverageMeter, softmax_temperature
import shutil
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--device', help='GPU device id to use [0]', default='cuda:0', type=str)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.', default=25, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.', default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.', default=0.001, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.', default='', type=str)
    parser.add_argument('--train_filename', dest='filename_list', help='Path to text file containing relative paths for every example.', default='', type=str)
    parser.add_argument('--test_filename', help='Path to textfile containing test images.', default='', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = 'cont', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.', default='', type=str)
    parser.add_argument('--size', type=int, default=112, help='Input size image'),
    parser.add_argument('--log_dir', type=str, default='log', help='log file direction')
    parser.add_argument('--tensorboardx_logdir', type=str, default='log/resnet50', help='tsboardx direction')
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    tensorboardx_logdir = os.path.join(args.log_dir, args.tensorboardx_logdir)
    if os.path.exists(tensorboardx_logdir):
        shutil.rmtree(tensorboardx_logdir)

    return args
args = parse_args()
lr = args.lr
num_epochs = args.num_epochs
batch_size = args.batch_size
device = args.device
data_dir = args.data_dir
train_filename = args.train_filename
test_filename = args.test_filename
total_dataset = FaceMaskDataset(data_dir, train_filename, True)
test_dataset  = FaceMaskDataset(data_dir, test_filename, False)

train_size = int(0.8 * len(total_dataset))
val_size = len(total_dataset) - train_size

train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss().to(device)

model = Res2Net(Bottle2neck, [3, 4, 6, 3])

summary(model, (3, 112, 112))
model = model.to(device)
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict and 'fc' not in k}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

load_filtered_state_dict(model, torch.load('../res2net50_v1b_26w_4s-3cf99910.pth'))
optimizer = torch.optim.Adam([{'params':model.parameters()}], lr = lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                        T_max=num_epochs*len(train_loader), eta_min=lr/10)

def accuracy(pred, labels):
    pred = softmax_temperature(pred, 1)
    _, pred = pred.topk(1,1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    res = []
    correct_k = correct[0].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]


def train(model, train_loader, epoch):
    model.train()

    train_loss = AverageMeter()
    train_acc = AverageMeter()

    for i, (imgs, labels) in enumerate(train_loader):
        if epoch == 0:
            optimizer.param_groups[0]['lr'] = (i*1.0/len(train_loader)*lr)
        imgs = imgs.to(device)
        labels = labels.to(device)

        pred = model(imgs)
        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)        
        
        optimizer.step() 
        if epoch != 0:
            lr_scheduler.step()

        train_loss.update(loss.item())
        acc = accuracy(pred, labels)
        train_acc.update(acc.item())

    print("Epoch: {epoch} Train Loss: {train_loss.avg:.4f}\t Accuracy: {acc.avg:.4f}\t Lr: {lr:.6f}".format(epoch=epoch, train_loss=train_loss, acc=train_acc, lr = optimizer.param_groups[0]['lr']))
    return train_loss.avg, train_acc.avg

def validate(model, val_loader):
    model.eval()

    val_loss = AverageMeter()
    val_acc = AverageMeter()

    for i, (imgs, labels) in enumerate(val_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            pred = model(imgs)
        loss = criterion(pred, labels)
        val_loss.update(loss.item())
        acc = accuracy(pred, labels)
        val_acc.update(acc.item())
    print("Loss: {val_loss.avg:.4f}\t Accuracy: {acc.avg:.4f}".format(val_loss=val_loss, acc=val_acc))
    return val_loss.avg, val_acc.avg

# validate(model, test_loader)
# sys.exit()
print(criterion)

best_val = np.inf
print("------------- Starting -----------")
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, epoch)
    val_loss, val_acc = validate(model, val_loader)
    test_loss, test_acc = validate(model, test_loader)
    if val_loss < best_val:
        best_val = val_loss
        print("Taking snapshot...")
        torch.save(model.state_dict(), os.path.join('snapshot', args.output_string))
        cnt = 0
    if val_loss > best_val:
        cnt += 1
        print("Epoch since last improvement ", cnt)
    if cnt == 10:
        print("Early stopping...\n")
        break
    



