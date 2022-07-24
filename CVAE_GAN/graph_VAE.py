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
import torch.utils.data as Data
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
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

    def forward(self, x):
        x = x.reshape(x.shape[0],-1)
        z = self.encoder(x)
        output = self.decoder(z)
        output = output.reshape(x.shape[0],-1)

        return output
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


class Discriminator(nn.Module):
    def __init__(self, outputn=1):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((1, 1)),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((1, 1)),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((1, 1)),

            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((1, 1)),

            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((1, 1)),

            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(40000, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, outputn),
            nn.Sigmoid()
        )

    def forward(self, input):
        # print('input:',input.shape)
        x = self.dis(input)
        x = x.view(x.size(0), -1)
        # print('x1:',x.shape)

        x = self.fc(x)
        return x.squeeze(1)


class Discriminator_C(nn.Module):
    def __init__(self, outputn=1):
        super(Discriminator_C, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((1, 1)),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((1, 1)),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),

            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((1, 1)),

            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((1, 1)),

            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((1, 1)),

            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(40000, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, outputn),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.dis(input)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        return x.squeeze(1)


def loss_function(recon_x, x, mean, logstd):
    # BCE = F.binary_cross_entropy(recon_x,x,reduction='sum')
    MSE = MSECriterion(recon_x, x)
    # 因为var是标准差的自然对数，先求自然对数然后平方转换成方差
    var = torch.pow(torch.exp(logstd), 2)
    KLD = -0.5 * torch.sum(1 + torch.log(var) - torch.pow(mean, 2) - var)
    return MSE + KLD

def load_data():
    data=np.load('img.npy', allow_pickle=True)
    data = torch.tensor(data)
    data = data.transpose(1,3).transpose(2,3)
    print(data.shape)
    return data

if __name__ == '__main__':
    # 设置初始化参数
    # dataset = dataset()
    fig = load_data()
    batchSize = 32
    # imageSize = 567
    nz = 7500
    # nz = 5120
    # nepoch = 1250
    nepoch = 10

    # ae_z = 64
    ae_z = 46

    if not os.path.exists('./img_CVAE-GAN'):
         os.mkdir('./img_CVAE-GAN')
    print("Random Seed: 42")
    random.seed(42)
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 可以优化运行效率
    cudnn.benchmark = True
    dataset = Data.TensorDataset(fig, fig)
    dataloader = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True)

    # 将模型,loss等导入cuda中
    print("=====> 构建VAE")
    vae = VAE().to(device)

    print("=====> 加载VAE")
    ae = AE().to(device)
    ae.load_state_dict(torch.load('AE.pth'))

    print("=====> 构建D")
    D = Discriminator(1).to(device)
    # D.load_state_dict(torch.load('./CVAE-GAN-Discriminator.pth'))
    print("=====> 构建C")
    C = Discriminator_C(46).to(device)
    # C.load_state_dict(torch.load('./CVAE-GAN-Classifier.pth'))
    criterion = nn.BCELoss().to(device)
    MSECriterion = nn.MSELoss().to(device)

    # 设置优化器参数
    print("=====> Setup optimizer")
    optimizerD = optim.Adam(D.parameters(), lr=0.0001)
    optimizerC = optim.Adam(C.parameters(), lr=0.0001)
    optimizerVAE = optim.Adam(vae.parameters(), lr=0.0001)

    d_loss_list = []
    c_loss_list = []
    g_loss_list = []
    epoch_list = []

    # 训练模型过程
    for epoch in range(nepoch):
        for i, (data, label) in enumerate(dataloader, 0):
            # 先处理一下数据
            data = data.float().to(device)
            label = label.float().to(device)
            # print('data shape:{},label shape:{}'.format(data.shape,label.shape))
            label = ae.encoder(label.reshape(label.shape[0], -1))
            label = label.detach()
            # print(label.shape)
            # 将label进行one-hot操作
            # label_onehot = torch.zeros((data.shape[0], 10)).to(device)
            # label_onehot[torch.arange(data.shape[0]), label] = 1
            batch_size = data.shape[0]

            # 先训练C
            output = C(data)
            real_label = label  # 定义真实的图片label
            errC = MSECriterion(output, real_label)
            C.zero_grad()
            errC.backward()
            optimizerC.step()


            # 再训练D
            # print('data shape:',data.shape)
            output = D(data)
            real_label = torch.ones(batch_size).to(device)  # 定义真实的图片label为1
            fake_label = torch.zeros(batch_size).to(device)  # 定义假的图片的label为0
            errD_real = criterion(output, real_label)

            z = torch.randn(batch_size, nz+label.shape[1]).to(device)  #(B,7500)
            # z = torch.randn(batch_size, nz+ 536).to(device)  #(B,7500)

            # print('z:',z.shape)
            fake_data = vae.decoder(z)
            # print('fake data shape:',fake_data.shape)

            output = D(fake_data)
            errD_fake = criterion(output, fake_label)

            errD = errD_real + errD_fake
            D.zero_grad()
            errD.backward()
            optimizerD.step()

            # 更新VAE(G)1
            z, mean, logstd = vae.encoder(data)
            z = torch.cat([z, label], 1)

            recon_data = vae.decoder(z)
            vae_loss1 = loss_function(recon_data, data, mean, logstd)

            # 更新VAE(G)2
            output = D(recon_data)
            real_label = torch.ones(batch_size).to(device)
            vae_loss2 = criterion(output, real_label)

            # 更新VAE(G)3
            output = C(recon_data)
            real_label = label

            vae_loss3 = MSECriterion(output, real_label)

            vae.zero_grad()
            vae_loss = vae_loss1 + vae_loss2 + vae_loss3
            vae_loss.backward()
            optimizerVAE.step()

            if i % 100 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_C: %.4f Loss_G: %.4f'
                      % (epoch, nepoch, i, len(dataloader),
                         errD.item(), errC.item(), vae_loss.item()))
            if epoch == 0:
                real_images = make_grid(data.cpu(), nrow=8, normalize=True).detach()
                save_image(real_images, 'img_CVAE-GAN/real_images.png')
            if i == len(dataloader) - 1:
                sample = torch.randn(data.shape[0], nz).to(device)
                # print('label:', label)
                sample = torch.cat([sample, real_label], 1)
                output = vae.decoder(sample)
                fake_images = make_grid(output.cpu(), nrow=8, normalize=True).detach()
                # print('img,', fake_images.shape)
                fake_images = transforms.ToPILImage()(fake_images)
                # temp = np.array(fake_images.getdata()).reshape(fake_images.size[0], fake_images.size[1], 3)
                fake_images.save('./img_CVAE-GAN/fake_images-{}.png'.format(epoch + 26))

        # 画出loss function的图
        if epoch % 10 == 0:
            d_loss_list.append(errD.item())
            c_loss_list.append(errC.item())
            g_loss_list.append(vae_loss.item())
            epoch_list.append(epoch)

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(epoch_list, c_loss_list, color="red")
            ax1.legend(loc=0)
            # 设置对应坐标轴的名称
            ax1.set_ylabel("C_loss")
            ax1.set_xlabel("epoch")

            ax2 = plt.twinx()
            ax2.set_ylabel("G_loss")
            ax2.plot(epoch_list, g_loss_list, color='blue')
            ax2.legend(loc=0)

            plt.savefig('C_G_loss_func_img.jpg')
            # plt.show()

            fig2 = plt.figure()
            plt.plot(epoch_list, d_loss_list)
            plt.ylabel('D_loss')
            plt.xlabel('epoch')
            plt.savefig('D_loss_func_img.jpg')
            # plt.show()

torch.save(vae.state_dict(), './CVAE-GAN-VAE.pth')
torch.save(D.state_dict(), './CVAE-GAN-Discriminator.pth')
torch.save(C.state_dict(), './CVAE-GAN-Classifier.pth')
print('over')
