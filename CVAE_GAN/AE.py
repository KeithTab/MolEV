import random
import torch.utils.data as Data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.nn import functional as F

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
        # print(x.shape)
        x = x.reshape(x.shape[0],-1)
        # print('input:',x.shape)
        z = self.encoder(x)
        # print(z.shape)
        output = self.decoder(z)
        output = output.reshape(x.shape[0],-1)
        # print('output:', output.shape)
        return output

def load_data():
    data=np.load('img.npy', allow_pickle=True)
    data = torch.tensor(data)
    data = data.transpose(1,3).transpose(2,3)
    print(data.shape)
    return data

def loss_function(recon_x, x, mu, logvar,BATCH_SIZE):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= BATCH_SIZE * 784
    return BCE + KLD

if __name__ == '__main__':
    # 设置初始化参数
    # dataset = dataset()
    fig = load_data()
    batchSize = 32
    imageSize = 567
    nz = 5120
    nepoch = 5
    # if not os.path.exists('./img_CVAE-GAN'):
    #     os.mkdir('./img_CVAE-GAN')
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
    ae = AE().to(device)

    # vae.load_state_dict(torch.load('./CVAE-GAN-VAE.pth'))
    print("=====> 构建D")
    # D = Discriminator(1).to(device)
    # D.load_state_dict(torch.load('./CVAE-GAN-Discriminator.pth'))
    print("=====> 构建C")
    # C = Discriminator(536).to(device)
    # C.load_state_dict(torch.load('./CVAE-GAN-Classifier.pth'))

    criterion = nn.MSELoss()
    # criterion = pytorch_ssim.SSIM()

    # 设置优化器参数
    print("=====> Setup optimizer")
    optimizerAE = optim.Adam(ae.parameters(), lr=0.0001)

    # 训练模型过程
    for epoch in range(nepoch):
        for data,label in dataloader:  # 载入数据
            # 先处理一下数据
            data = data.float().to(device)
            label = label.reshape(label.shape[0],-1).float().to(device)

            output = ae(data)
            # print(output.shape,data.shape)
            err = criterion(output, label)
            ae.zero_grad()
            err.backward()
            optimizerAE.step()

        print(epoch)
    torch.save(ae.state_dict(), 'AE.pth')
