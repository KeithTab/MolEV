import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from dataset_2 import dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from dataset_2 import load_label
from PIL import Image
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # 定义AE模型来解决label问题
        self.encoder = nn.Sequential(
            # nn.Linear(536, 256),
            nn.Linear(7500, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 46),
        )

        self.decoder = nn.Sequential(
            nn.Linear(46, 128),
            nn.ReLU(0.2),
            nn.Linear(128, 256),
            nn.ReLU(0.2),
            nn.Linear(256, 7500),
        )

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 定义编码器

        # self.en_shape = 32 * 4 * 3
        self.en_shape = 32 * 25 * 25

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # self.encoder_fc1 = nn.Linear(self.en_shape, nz)
        self.encoder_fc1 = nn.Linear(5408, nz)
        self.encoder_fc2 = nn.Linear(5408, nz)
        self.Sigmoid = nn.Sigmoid()
        # self.decoder_fc = nn.Linear(nz + 536, self.en_shape)
        self.decoder_fc = nn.Linear(7500+46, self.en_shape)
        # self.decoder_deconv = nn.Sequential(
        #     nn.ConvTranspose2d(32, 16, 4, 2, 1),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(16, 3, 4, 2, 1),
        #     # nn.Sigmoid(),
        # )
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def noise_reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)  # 根据论文中所述，加入随机变量，使得VAE模型更贴近于GAN模型
        z = mean + eps * torch.exp(logvar * 0.5)  # 将平均值和标准差组合成中间变量z
        return z

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output

    def encoder(self, x):
        # print(x.shape)
        out1, out2 = self.encoder_conv(x), self.encoder_conv(x)
        # print(out1.shape,out2.shape)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))  # 计算出VAE中的平均值参数
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))  # 计算出VAE中的std标准差参数
        z = self.noise_reparameterize(mean, logstd)
        # print(z.shape)
        return z, mean, logstd

    def decoder(self, z):
        # print('input z:',z.shape)
        out3 = self.decoder_fc(z)
        # print('out3:',out3.shape)
        out3 = out3.view(out3.shape[0], 32, 25, 25)
        # print('out3_1:',out3.shape)
        out3 = self.decoder_deconv(out3)
        # print('out3_result:',out3.shape)
        return out3


device = 'cuda' if torch.cuda.is_available() else 'cpu'
nz = 7500
batch = 4
label = load_label()

ae = AE().to(device)
# ae.load_state_dict(torch.load('./AE.pth'))
ae.load_state_dict(torch.load('AE.pth'))

vae = VAE().to(device)
# vae.load_state_dict(torch.load('./CVAE-GAN-VAE.pth'))
vae.load_state_dict(torch.load('CVAE-GAN-VAE.pth'))

# 使用真的label来制造图片
real_label = label[0].reshape([1, label.shape[1]])
# print(real_label.shape)
# real_label = torch.FloatTensor(real_label).to(device)
real_label = torch.tensor(real_label).float().to(device)

# print(real_label.shape)
sample = torch.randn(1, nz).to(device)
# print('label:', label)
sample = torch.cat([sample, real_label], 1)
# print(sample.shape)
output = vae.decoder(sample)[0]
fake_images = make_grid(output.cpu(), normalize=True).detach()
# print('img,', fake_images.shape)
fake_images = transforms.ToPILImage()(fake_images)

temp = np.array(fake_images.getdata()).reshape(fake_images.size[0], fake_images.size[1], 3)
# print(temp.shape)
# temp = temp.reshape(16, 12, 3)
temp = temp.reshape(50, 50, 3)
# print(temp.shape)

for i in range(16):
    for ii in range(12):
        if temp[i][ii][0] > 220 and temp[i][ii][1] > 220 and temp[i][ii][2] > 220:
            temp[i][ii] = [255, 255, 255]
        if ii > 8:
            temp[i][ii] = [255, 255, 255]

# print(temp)


fake_images = Image.fromarray(temp.reshape(50, 50, 3).astype('uint8')).convert('RGB')
# fake_images = Image.fromarray(temp.reshape(16, 12, 3).astype('uint8')).convert('RGB')


fake_images.save('./img_CVAE-GAN_TEST/fake_images-{}.png'.format(0))

# 使用AE制造假的label来输入图片
fake_label = torch.randn(1, 46).to(device)
# fake_label = ae.decoder(fake_label)
# print(fake_label.shape)
# fake_label = ae.decoder(fake_label)
# fake_label = fake_label.detach()
sample = torch.randn(1, nz).to(device)
# print('label:', label)

print(sample.shape,fake_label.shape)
sample = torch.cat([sample, fake_label], 1)

print(sample.shape)
output = vae.decoder(sample)[0]
fake_images = make_grid(output.cpu(), normalize=True).detach()
# print('img,', fake_images.shape)
fake_images = transforms.ToPILImage()(fake_images)

temp = np.array(fake_images.getdata()).reshape(fake_images.size[0], fake_images.size[1], 3)
# print(temp.shape)

# temp = temp.reshape(16, 12, 3)
temp = temp.reshape(50, 50, 3)

for i in range(16):
    for ii in range(12):
        if temp[i][ii][0] > 200 and temp[i][ii][1] > 200 and temp[i][ii][2] > 200:
            temp[i][ii] = [255, 255, 255]
        if ii > 8:
            temp[i][ii] = [255, 255, 255]

# print(temp)


fake_images = Image.fromarray(temp.astype('uint8')).convert('RGB')

fake_images.save('./img_CVAE-GAN_TEST/fake_images-{}.png'.format(1))
