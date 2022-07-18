import os
import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image, ImageFilter
import logging as logger
import utils
import random
def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines
def is_blur(img):
    """Decide if the input image is blur or not
    :param img: input image
    :type img: numpy.array
    :returns True if image is blur, return blury as well
    """    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplace = cv2.Laplacian(img, cv2.CV_64F)
    blury = laplace.var()
    
    return blury
def brightness(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_img)

def transform(image):
    """ Transform a image by cv2.
    """
    img_size = image.shape[0]
    #adjust brightness and contrast
    if random.random() > 0.5:
        alpha = 0.75 + 0.5*random.random()
        beta = np.random.choice(np.arange(-15, 15, 2), 1)[0]
        if alpha < 1.05 and beta < 0:
            if brightness(image) > 200:
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        else:
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    #center crop and resize 112x112
    if is_blur(image) > 120:
        if random.random() > 0.5:
            rand_crop = np.random.choice(np.arange(5, 11), 1)[0]
            crop_image = image[rand_crop:-rand_crop, rand_crop:-rand_crop]
            image = cv2.resize(crop_image, (112, 112))
    
    # random crop
    if is_blur(image) > 120:
        if random.random() > 0.3:
            new_sz = np.random.choice(np.arange(30, 100, 3), 1)[0]
            image = cv2.resize(image,(new_sz,new_sz))
            image = cv2.resize(image, (img_size,img_size))
        
    #Gaussian blur
    if is_blur(image) > 120:
        if random.random() > 0.6:
            sz = np.random.choice(np.arange(3, 10, 2), 1)[0]
            image = cv2.GaussianBlur(image, (sz, sz), 0)
            # image = cv2.GaussianBlur(image, (7,7), 0)
        else:
            if random.random() > 0.3:
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                image = cv2.filter2D(image, -1, kernel)

    if is_blur(image) > 120:
        if random.random() > 0.5:
            ps = np.random.choice(np.arange(2, 6), 1)[0]
            resize_img = cv2.resize(image, (112 - ps*2, 112 - ps*2))
            image = cv2.copyMakeBorder(resize_img.copy(), ps, ps, ps, ps, cv2.BORDER_CONSTANT)

    if random.random() > 0.7:
        image= cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    # normalizing
    if image.ndim == 2:
        image = (image - 127.5) * 0.0078125
        new_image = np.zeros([3,img_size,img_size], np.float32)
        new_image[0,:,:] = image
        image = torch.from_numpy(new_image.astype(np.float32))
    else:
        image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
        image = torch.from_numpy(image.astype(np.float32))
    return image
    
class Pose_300W_LPA_60bin(Dataset):
    # Head pose from 300W-LP dataset
    def __init__(self, data_dir, filename_path, trans=True, image_mode='RGB'):
        self.data_dir = data_dir
        filename_list = get_list_from_filenames(filename_path)
        self.transform = trans
        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):        
        img = cv2.imread((os.path.join(self.data_dir, self.X_train[index])))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # LFPW_image_train_0170_12_-0.597_1.359_-0.345.jpg
        img_name = os.path.basename(self.X_train[index])
        pose = np.array(img_name[0:-4].split("_")[-3:]).astype("float32")
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            yaw = -yaw
            roll = -roll
        if self.transform == True:
            img = transform(img)
        
        # Bin values
        bins = np.array(range(-93, 96, 3))
        # print(len(bins))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 122,450
        return self.length

class FaceMaskDataset(Dataset):
    def __init__(self, data_dir, filename, trans):
        super().__init__()
        self.data_dir = data_dir
        self.filename = filename
        self.trans = trans
        self.file_list = get_list_from_filenames(filename)
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, index):
        img_path = self.file_list[index]
        temp = img_path.split('/')[-2]
        if temp == 'unmask':
            label = 0 #torch.LongTensor([0])
        else:
            label = 1 #torch.LongTensor([1])
        img = cv2.imread(img_path)
        if self.trans:
            img = transform(img)
        else:
            img = (img.transpose((2, 0, 1)) - 127.5) * 0.0078125
            img = torch.from_numpy(img.astype(np.float32))
        return img, label

class D4KD(Dataset):
    # joint face mask and head pose dataset
    def __init__(self, data_dir, filename_path, trans=True, image_mode='RGB'):
        self.data_dir = data_dir
        filename_list = get_list_from_filenames(filename_path)
        self.transform = trans
        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):        
        img = cv2.imread((os.path.join(self.data_dir, self.X_train[index])))
        img_name = os.path.basename(self.X_train[index])
        
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        if self.transform == True:
            img = transform(img)

        return img, self.X_train[index]

    def __len__(self):
        # 122,450
        return self.length

class AFLW2000_custom(Dataset):
    def __init__(self, data_dir, filename_path, reLabel=True, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        self.reLabel = reLabel

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        link_img = os.path.join(self.data_dir, self.X_train[index] + self.img_ext)
        assert os.path.isfile(link_img) == True
        
        mat_path = os.path.join(self.data_dir, os.path.basename(self.y_train[index]) + self.annot_ext)

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        image = cv2.imread(link_img)
        image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
        image = torch.from_numpy(image.astype(np.float32))

        return image, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 2,000
        return self.length
class BIWI_60bin(Dataset):
    def __init__(self, data_dir, filename_path):
        self.data_dir = data_dir
        self.filename_path = filename_path
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.length = len(filename_list)
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.X_train[index])
        txt_path = img_path[0:-7] + 'pose.txt'
        pose_annot = open(txt_path, "r")
        R = []
        for line in pose_annot:
            line = line.strip('\n').split(' ')
            l = []
            if line[0] != '':
                for nb in line:
                    if nb == '':
                        continue
                    l.append(float(nb))
                R.append(l)

        R = np.array(R)
        T = R[3,:]
        R = R[:3,:]
        pose_annot.close()

        R = np.transpose(R)

        roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
        yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
        pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi
        bins = np.array(range(-93, 96, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
        image = torch.from_numpy(image.astype(np.float32))

        return image, labels, cont_labels, self.X_train[index]
    
    def __len__(self):
        return self.length
