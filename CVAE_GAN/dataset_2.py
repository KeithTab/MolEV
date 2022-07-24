import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import torch.nn as nn
from PIL import Image


# 导入所有的图片文件路径
def load_img():
    x = np.load('data/img.npy')
    return x


# 导入所有的label
def load_label():
    x_1 = np.load('data/best_A_atomtypes.npy')
    x_2 = np.load('data/best_A_atomxyz.npy')
    x_3 = np.load('data/best_A_predcoords.npy')
    x_4 = np.load('data/best_A_predfeatures_emb1.npy')
    # x_1 = np.load('data/new1aki_A_atomtypes.npy')
    # x_2 = np.load('data/new1aki_A_atomxyz.npy')
    # x_3 = np.load('data/new1aki_A_predcoords.npy')
    # x_4 = np.load('data/new1aki_A_predfeatures_emb1.npy')
    print(x_1.shape,x_2.shape,x_3.shape,x_4.shape)
    y = np.concatenate((x_1[:996], x_2[:996], x_3[:996], x_4[:996]), axis=1)
    # print(y.shape)
    return y


# 创建数据库类
class colormap(Dataset):
    def __init__(self, img_list, label_list, transform=None):
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = self.img_list[idx]  # load img
        label = self.label_list[idx]  # load landmark
        sample = {'image': image, 'label': label}  # encapsulate data
        # image = np.reshape(image, [3, 16, 12])
        image = Image.fromarray(image.astype('uint8')).convert('RGB')
        image = transforms.PILToTensor()(image)
        image = image.float()
        # print('image:',image.shape)
        label = torch.FloatTensor(label)
        # label = label.unsqueeze(0)
        return image, label


# 将数据库类实例化，即对图片，标签等使用重构化的数据库类
def dataset():
    img = load_img()
    # print(img.shape)
    y = load_label()
    # print(img_path)
    data = colormap(img, y, transform=transforms.ToTensor())
    # print(type(data[0][1]))
    return data


# test
if __name__ == '__main__':
    data = dataset()
    train_loader = DataLoader(dataset=data, batch_size=32, shuffle=False)
    print()
    for i, (data, label) in enumerate(train_loader):
        # x = torch.cat([data, label], 3)
        print(i)
        print(data[0][0][0])
        temp = data[0]
        print(data[0].shape)
        print(label.shape)
