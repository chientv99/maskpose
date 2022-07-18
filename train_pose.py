import sys, os, argparse, time
import numpy as np
import cv2
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
# from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
# import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import datasets
import torch.utils.model_zoo as model_zoo
import utils
from utils import AverageMeter
import logging as logger
# from backbone.backbone_def import BackboneFactory
from torchvision.models.resnet import Bottleneck, BasicBlock
from models import BoTNet50_2, ResNet_60bin, Res2Net, Bottle2neck
# from tensorboardX import SummaryWriter
import shutil
from torchsummary import summary
from torch.optim.swa_utils import AveragedModel, SWALR 
from torch.optim.lr_scheduler import CosineAnnealingLR

logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--device', help='GPU device id to use [0]', default='cuda:0', type=str)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.', default=25, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.', default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.', default=0.001, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.', default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.', default='', type=str)
    parser.add_argument('--filename_list2000', help='Path to textfile containing test images.', default='filename_path_2000.txt', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = 'cont', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',  default=1., type=float)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.', default='', type=str)
    parser.add_argument('--size', type=int, default=112, help='Input size image'),
    parser.add_argument('--log_dir', type=str, default='log', help='log file direction')
    parser.add_argument('--tensorboardx_logdir', type=str, default='log/resnet50', help='tsboardx direction')
    parser.add_argument('--filename_biwi', dest='filename_biwi', default="", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    tensorboardx_logdir = os.path.join(args.log_dir, args.tensorboardx_logdir)
    if os.path.exists(tensorboardx_logdir):
        shutil.rmtree(tensorboardx_logdir)
    # writer = SummaryWriter(log_dir=tensorboardx_logdir)
    
    # args.writer = writer
    return args

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

def compute_loss(pose, pred, labels, cont_labels, idx_tensor, criterion, softmax, alpha, reg_criterion):
    if pose == 'yaw':
        dim = 0
    elif pose == 'pitch':
        dim = 1
    elif pose == 'roll':
        dim =2
    else:
        raise IndexError("{} is not in ['yaw','pitch','roll']".format(pose))
    # Binned labels
    label = labels[:,dim]
    # # Continuous labels
    label_cont = cont_labels[:,dim]
    # # Cross entropy loss
    loss_cls = criterion(pred, label)
    # # MSE loss
    predicted = softmax(pred)
    predicted = torch.sum(predicted * idx_tensor, 1) * 3 - 93
    loss_reg = reg_criterion(predicted, label_cont)

    # Total loss
    loss = loss_cls + alpha * loss_reg
    return loss

def compute_error(axis, cont_labels, pred, idx_tensor, device=torch.device('cuda:0')):
    if axis == 'yaw':
        dim = 0
    elif axis == 'pitch':
        dim = 1
    elif axis == 'roll':
        dim = 2
    else:
        raise IndexError("{} not in ['yaw', 'pitch', 'roll']".format(axis))

    label = cont_labels[:,dim].to(device)
    # Continuous predictions
    predicted = utils.softmax_temperature(pred.data, 1)    

    predicted = torch.sum(predicted * idx_tensor, 1).to(device) * 3 - 93

    # Mean absolute error
    error = torch.sum(torch.abs(predicted - label))
    return error

def train(train_loader, idx_tensor, criterion, softmax, alpha, reg_criterion, optimizer, epoch, num_epochs, lr_scheduler):
    model.train()
    loss_yaw_meter = AverageMeter()
    loss_pitch_meter = AverageMeter()
    loss_roll_meter = AverageMeter() 
    for i, (images, labels, cont_labels, name) in enumerate(train_loader):     
        if epoch == 0:
            optimizer.param_groups[0]['lr'] = (i*1.0/len(train_loader)*args.lr)
     
        images = images.to(device)
        labels = labels.to(device)
        cont_labels = cont_labels.to(device)   

        yaw, pitch, roll = model(images)
        
        loss_yaw = compute_loss('yaw', yaw, labels, cont_labels, idx_tensor, criterion, softmax, alpha, reg_criterion)
        loss_pitch = compute_loss('pitch', pitch, labels, cont_labels, idx_tensor, criterion, softmax, alpha, reg_criterion)
        loss_roll = compute_loss('roll', roll, labels, cont_labels, idx_tensor, criterion, softmax, alpha, reg_criterion)
        
        loss_seq = loss_yaw + loss_pitch + loss_roll
        optimizer.zero_grad()
        loss_seq.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5) 
        optimizer.step() 
        if (epoch != 0):
            lr_scheduler.step()
        loss_yaw_meter.update(loss_yaw.data.item())
        loss_pitch_meter.update(loss_pitch.data.item())
        loss_roll_meter.update(loss_roll.data.item())
        loss_val = (loss_yaw_meter.val + loss_pitch_meter.val + loss_roll_meter.val)/3
        loss_avg = (loss_yaw_meter.avg + loss_pitch_meter.avg + loss_roll_meter.avg)/3

        if i % int(len(train_loader)/3) == 0:
            logger.info('Epoch [%d/%d], Iter [%d/%d], Losses: Yaw %.4f, Pitch %.4f, Roll %.4f, Total %.4f, Lr %.7f'
                    %(epoch+1, num_epochs, i, len(train_loader), loss_yaw_meter.val, loss_pitch_meter.val, 
                                                loss_roll_meter.val, loss_val, optimizer.param_groups[0]['lr']))
        if i+1 == len(train_loader):
            logger.info('Epoch [%d/%d], Iter [%d/%d], Losses_avg: Yaw %.4f, Pitch %.4f, Roll %.4f, Total %.4f, Lr %.7f'
                    %(epoch+1, num_epochs, i, len(train_loader), loss_yaw_meter.avg, loss_pitch_meter.avg, 
                                                loss_roll_meter.avg, loss_avg, optimizer.param_groups[0]['lr']))
    return loss_yaw_meter.avg

def validate(val_loader, model, idx_tensor, epoch):
    model.eval()
    total, yaw_error, pitch_error, roll_error = 0.,0.,0.,0.
    
    for i, (images, labels, cont_labels, name) in enumerate(val_loader):        
        with torch.no_grad():
            images = images.to(device)
            cont_labels = cont_labels.to(device)
            total += cont_labels.shape[0]
            #Forward pass
            yaw, pitch, roll = model(images)
            yaw_error += compute_error('yaw', cont_labels, yaw, idx_tensor, device)
            pitch_error += compute_error('pitch', cont_labels, pitch, idx_tensor, device)
            roll_error += compute_error('roll', cont_labels, roll, idx_tensor, device)
            total_error = (yaw_error + pitch_error + roll_error)/3
    logger.info('Valid (or Test) error in degrees '+str(total)+ ' valid images:\n'
                +'Yaw: {:.4f}, Pitch {:.4f}, Roll {:.4f}, Total {:.4f}'.format(yaw_error/total, pitch_error/total, roll_error/total, total_error/total))

    return total_error/total

if __name__ == '__main__':
    args = parse_args()
    device = args.device
    # cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    # gpu = args.gpu_id

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')
        
    model = ResNet_60bin(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    summary(model, (3, 112, 112))
    print('Model: ResNet152')
    
    if args.snapshot == 'load':
        load_filtered_state_dict(model, model_zoo.load_url(model_urls['resnet18']))
        print('Loaded model')
    else:
        print('Loaded from snapshot')
        saved_state_dict = torch.load(args.snapshot)
        # model.load_state_dict(saved_state_dict)
        load_filtered_state_dict(model, saved_state_dict)

    print('Loading data.')

    pose_dataset = datasets.Pose_300W_LPA_60bin(args.data_dir, args.filename_list)

    aflw2000_dataset = datasets.AFLW2000_custom(args.data_dir, args.filename_list2000, reLabel=True)
    biwi_dataset = datasets.BIWI_60bin(args.data_dir, args.filename_biwi)
    
    train_size = int(0.8*len(pose_dataset))
    val_size = len(pose_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(pose_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=2)

    biwi_loader = torch.utils.data.DataLoader(dataset=biwi_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(len(train_loader))
    model.to(device)

    print("Criterion: Cross Entropy loss")
    criterion = nn.CrossEntropyLoss().to(device)
    # reg_criterion = nn.MSELoss(reduce=False).to(device)
    reg_criterion = nn.MSELoss().to(device)
    # Regression loss coefficient
    alpha = torch.tensor(args.alpha).to(device)

    softmax = nn.Softmax(dim=1).to(device)
    idx_tensor = [idx for idx in range(62)]
    idx_tensor = torch.Tensor(idx_tensor).to(device)

    optimizer = torch.optim.Adam([{'params':model.parameters()}], lr = args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                        T_max=args.num_epochs*len(train_loader), eta_min=args.lr/1000)
    logger.info('Ready to train network.')
    best_err = 100.
    best_test_err = 100.
    epoch_since_last_best_ValError = 0

    for epoch in range(num_epochs):
        loss_train = train(train_loader, idx_tensor, criterion, softmax, alpha, reg_criterion, optimizer, epoch, num_epochs, lr_scheduler)
        print('Validating..')
        error = validate(val_loader, model, idx_tensor, epoch)
        print('Testing..')
        test_error = validate(biwi_loader, model, idx_tensor, epoch)
        
        if error < best_err:
            best_err = error
            print('Taking snapshot as Best_' + args.output_string + '.pkl\n')#+ '_epoch_'+ str(epoch+1)
            torch.save(model.state_dict(),'output/snapshots/' + args.output_string + '_best.pkl')#+ '_epoch_'+ str(epoch+1)
        
        torch.save(model.state_dict(), 'output/snapshots/' + args.output_string + '_final.pkl')