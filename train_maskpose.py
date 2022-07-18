import torch
from torch import nn
import math
import torchvision
import torch.utils.model_zoo as model_zoo
# from torchsummary import summary
from models import MergeResNet, ResNet
import torch.nn.functional as F
import utils
from utils import AverageMeter
# import logging as logger
import shutil
import sys
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchsummary import summary
from tqdm import tqdm
import os
import argparse
sys.path.append('../')
import datasets
from models import Res2Net, Bottle2neck, BoTNet50_2, ResNet_60bin

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--device', help='GPU device id to use [0]', default='cuda:0', type=str)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.', default=25, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.', default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.', default=0.001, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.', default='', type=str)
    parser.add_argument('--maskpose_filename', dest='filename_list', help='', default='', type=str)
    parser.add_argument('--lpa_filename', dest='filename_list', help='', default='', type=str)
    parser.add_argument('--aflw2000_filename', dest='filename_list', help='', default='', type=str)
    parser.add_argument('--biwi_filename', dest='filename_list', help='', default='', type=str)
    parser.add_argument('--mask_filename', dest='filename_list', help='', default='', type=str)
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

device = args.device
batch_size = args.batch_size
epochs = args.num_epochs
init_lr = args.lr

alpha = 2
T = 3
def accuracy(pred, labels):
    pred = utils.softmax_temperature(pred, 1)
    _, pred = pred.topk(1,1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    res = []
    correct_k = correct[0].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]

def load_filtered_state_dict(model, state_dict):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    print('Loaded', len(state_dict.keys()), 'layers.')
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

expected_model = MergeResNet(BasicBlock, [2, 2, 2, 2])
snapshot = "/snapshots/Pose_R18.pkl"
state_dict = torch.load(snapshot)
load_filtered_state_dict(expected_model, state_dict)

#load pose teacher
model1 = BoTNet50_2(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3])
snapshot1 = torch.load("/snapshots/Pose_BoTNet101.pkl")
load_filtered_state_dict(model1, snapshot1)
model1.to(device)
model1.eval()

model2 = Res2Net(Bottle2neck, [3, 4, 23, 3])
snapshot2 = torch.load("/snapshots/Pose_Res2Net101.pkl")
load_filtered_state_dict(model2, snapshot2)
model2.to(device)
model2.eval()

model3 = ResNet_60bin(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3])
snapshot = torch.load("/snapshots/Pose_ResNet101.pkl")
load_filtered_state_dict(model3, snapshot)
model3.to(device)
model3.eval()

#load mask teacher
mask_model = ResNet(Bottleneck, [3, 8, 36, 3])
mask_snapshot = "/snapshot/tMask_ResNet152.pt"
mask_state_dict = torch.load(mask_snapshot)
load_filtered_state_dict(mask_model, mask_state_dict)
mask_model.to(device)
mask_model.eval()


pose_dataset = datasets.D4KD("", args.maskpose_filename)
p300wlpa = datasets.Pose_300W_LPA_60bin("", args.lpa_filename)
aflw2000_dataset = datasets.AFLW2000_custom("", args.aflw2000_filename, reLabel=True)
biwi_dataset = datasets.BIWI_60bin('', args.biwi_filename)
test_mask_dataset = datasets.FaceMaskDataset("", args.mask_filename, trans=False)

train_size = int(0.8*len(test_mask_dataset))
val_size = len(test_mask_dataset) - train_size
mtrain_dataset, mval_dataset = torch.utils.data.random_split(test_mask_dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(dataset=pose_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
aflw_loader  = torch.utils.data.DataLoader(dataset=aflw2000_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
biwi_loader = torch.utils.data.DataLoader(dataset=biwi_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_mask_loader = torch.utils.data.DataLoader(dataset=mtrain_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_mask_loader2 = torch.utils.data.DataLoader(dataset=mval_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
p300wlpa_loader = torch.utils.data.DataLoader(dataset=p300wlpa, batch_size=batch_size, shuffle=False, num_workers=2)


criterion = nn.CrossEntropyLoss().to(device)
reg_criterion = nn.MSELoss().to(device)
softmax = nn.Softmax(dim=1).to(device)
idx_tensor = [idx for idx in range(62)]
idx_tensor = torch.Tensor(idx_tensor).to(device)

def compute_kd_loss(student, teacher1, teacher2, teacher3):
    teacher1 = F.softmax(teacher1, dim=1)
    teacher2 = F.softmax(teacher2, dim=1)
    teacher3 = F.softmax(teacher3, dim=1)
    teacher = (teacher1 + teacher2 + teacher3) / 3
    KD_loss = nn.KLDivLoss()(F.log_softmax(student, dim=1), teacher)
    return KD_loss

def loss_fn_kd(outputs, teacher_outputs):
    T = 3
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) 

    return KD_loss

def mae(axis, cont_labels, pred, idx_tensor, device):
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

def train(train_loader, epoch):
    expected_model.train()
    loss_yaw_meter  = AverageMeter()
    loss_pitch_meter = AverageMeter()
    loss_roll_meter = AverageMeter()
    loss_mask_meter = AverageMeter()
    for i, (images, name) in enumerate(train_loader):
        if epoch == 0:
            optimizer.param_groups[0]['lr'] = (i*1.0/len(train_loader)*init_lr)
     
        images = images.to(device)

        mask, yaw, pitch, roll = expected_model(images)
        with torch.no_grad():
            yaw1, pitch1, roll1 = model1(images)
            yaw2, pitch2, roll2 = model2(images)
            yaw3, pitch3, roll3 = model3(images)
            mask_teacher = mask_model(images)

        loss_yaw = compute_kd_loss(yaw, yaw1, yaw2, yaw3)
        loss_pitch = compute_kd_loss(pitch, pitch1, pitch2, pitch3)
        loss_roll = compute_kd_loss(roll, roll1, roll2, roll3)

        loss_mask = loss_fn_kd(mask, mask_teacher)
        loss = loss_mask + (loss_yaw + loss_pitch + loss_roll)*10
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(expected_model.parameters(), clip_value=0.5)
        optimizer.step()

        if epoch != 0:
            lr_scheduler.step()
        
        loss_yaw_meter.update(loss_yaw.data.item())
        loss_pitch_meter.update(loss_pitch.data.item())
        loss_roll_meter.update(loss_roll.data.item())
        loss_mask_meter.update(loss_mask.data.item())

        loss_val = (loss_yaw_meter.val + loss_pitch_meter.val + loss_roll_meter.val)/3
        loss_avg = (loss_yaw_meter.avg + loss_pitch_meter.avg + loss_roll_meter.avg)/3

        if i % int(len(train_loader)/3) == 0:
            print('Epoch [%d/%d], Iter [%d/%d], KDLosses: Mask %.4f, Yaw %.4f, Pitch %.4f, Roll %.4f, Total %.4f, Lr %.7f'
                    %(epoch+1, epochs, i, len(train_loader), loss_mask_meter.val, loss_yaw_meter.val, loss_pitch_meter.val, 
                                                loss_roll_meter.val, loss_val, optimizer.param_groups[0]['lr']))
        if i+1 == len(train_loader):
            print('Epoch [%d/%d], Iter [%d/%d], KDLosses_avg: Mask %.4f, Yaw %.4f, Pitch %.4f, Roll %.4f, Total %.4f, Lr %.7f'
                    %(epoch+1, epochs, i, len(train_loader), loss_mask_meter.avg, loss_yaw_meter.avg, loss_pitch_meter.avg, 
                                                loss_roll_meter.avg, loss_avg, optimizer.param_groups[0]['lr']))
    return loss_val
def pose_validate(val_loader, student_model):
    student_model.eval()
    total, yaw_err, pitch_err, roll_err = 0.,0.,0.,0.
    for i, (images, labels, cont_labels, name) in tqdm(enumerate(val_loader)):
        with torch.no_grad():
            images = images.to(device)
            cont_labels = cont_labels.to(device)
            total += cont_labels.shape[0]
            yaw, pitch, roll = student_model(images)
            yaw_err += mae('yaw', cont_labels, yaw, idx_tensor, device)
            pitch_err += mae('pitch', cont_labels, pitch, idx_tensor, device)
            roll_err += mae('roll', cont_labels, roll, idx_tensor, device)
    total_err = (yaw_err + pitch_err + roll_err) / 3 
    print('Valid (or Test) error in degrees '+str(total)+ ' valid images:\n'
                +'Yaw: {:.4f}, Pitch {:.4f}, Roll {:.4f}, Total {:.4f}'.format(yaw_err/total, pitch_err/total, roll_err/total, total_err/total))
    return total_err / total
def mpose_validate(val_loader, student_model):
    student_model.eval()
    total, yaw_err, pitch_err, roll_err = 0.,0.,0.,0.
    for i, (images, labels, cont_labels, name) in enumerate(val_loader):
        with torch.no_grad():
            images = images.to(device)
            cont_labels = cont_labels.to(device)
            total += cont_labels.shape[0]
            mask, yaw, pitch, roll = student_model(images)
            yaw_err += mae('yaw', cont_labels, yaw, idx_tensor, device)
            pitch_err += mae('pitch', cont_labels, pitch, idx_tensor, device)
            roll_err += mae('roll', cont_labels, roll, idx_tensor, device)
    total_err = (yaw_err + pitch_err + roll_err) / 3 
    print('Valid (or Test) error in degrees '+str(total)+ ' valid images:\n'
                +'Yaw: {:.4f}, Pitch {:.4f}, Roll {:.4f}, Total {:.4f}'.format(yaw_err/total, pitch_err/total, roll_err/total, total_err/total))
    return total_err / total

def mask_validate(val_loader, student_model):
    student_model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()

    for i, (imgs, labels) in enumerate(val_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        pred, yaw, pitch, roll = student_model(imgs)
        loss = criterion(pred, labels)
        val_loss.update(loss.item())
        acc = accuracy(pred, labels)
        val_acc.update(acc.item())
    print("Loss: {val_loss.avg:.4f}\t Accuracy: {acc.avg:.4f}".format(val_loss=val_loss, acc=val_acc))
    return val_loss.avg
optimizer = torch.optim.Adam([{'params': expected_model.parameters()}], lr = init_lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader), eta_min=init_lr/100)
saved_snapshot = os.path.join('snapshot', args.output_string)
print('Ready to train network')
best_err = 100.



# sys.exit()
for epoch in range(epochs):
    loss_train = train(train_loader, epoch)
    val_pose_err = pose_validate(biwi_loader, expected_model)
    val_mask_err = mask_validate(test_mask_loader, expected_model)
    test_err = val_pose_err + val_mask_err
    if test_err < best_err:
        best_err = test_err
        print("Taking snapshot...")
        torch.save(expected_model.state_dict(), saved_snapshot)