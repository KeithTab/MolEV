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


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # 定义AE模型来解决label问题
        self.encoder = nn.Sequential(
            nn.Linear(536, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(0.2),
            nn.Linear(128, 256),
            nn.ReLU(0.2),
            nn.Linear(256, 536),
        )

    # def encoder(self, x):
    #     output = self.encoder(x)
    #     return output
    #
    # def decoder(self, x):
    #     output = self.decoder(x)
    #     return output

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output


if __name__ == '__main__':
    # 设置初始化参数
    dataset = dataset()
    batchSize = 32
    imageSize = 567
    nz = 5120
    nepoch = 128
    # if not os.path.exists('./img_CVAE-GAN'):
    #     os.mkdir('./img_CVAE-GAN')
    print("Random Seed: 42")
    random.seed(42)
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 可以优化运行效率
    cudnn.benchmark = True
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

    MSECriterion = nn.MSELoss().to(device)

    # 设置优化器参数
    print("=====> Setup optimizer")
    optimizerAE = optim.Adam(ae.parameters(), lr=0.0001)

    # 训练模型过程
    for epoch in range(nepoch):
        for i, (data, label) in enumerate(dataloader, 0):
            # 先处理一下数据
            data = data.to(device)
            label = label.to(device)

            output = ae(label)
            err = MSECriterion(output, label)
            ae.zero_grad()
            err.backward()
            optimizerAE.step()

        print(epoch)
    torch.save(ae.state_dict(), './AE.pth')
