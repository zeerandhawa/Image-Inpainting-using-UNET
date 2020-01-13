import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.downStep1 = downStep(4, 64)
        self.downStep2 = downStep(64, 128)
        self.downStep3 = downStep(128, 256)
        self.downStep4 = downStep(256, 512)
        self.downStep5 = downStep(512, 1024)

        self.pool = maxPool()

        self.upStep1 = upStep(1024, 512)
        self.upStep2 = upStep(512, 256)
        self.upStep3 = upStep(256, 128)
        self.upStep4 = upStep(128, 64, withReLU = True)
        self.outputLayer = outputLayer(64, 3)

    def forward(self, x):
        # todo
        x1 = self.downStep1(x)

        x = self.pool(x1)
        x2 = self.downStep2(x)

        x = self.pool(x2)
        x3 = self.downStep3(x)

        x = self.pool(x3)
        x4 = self.downStep4(x)

        x = self.pool(x4)
        x = self.downStep5(x)

        x = self.upStep1(x, x4)

        x = self.upStep2(x, x3)
        x = self.upStep3(x, x2)
        x = self.upStep4(x, x1)
        x = self.outputLayer(x)

        return x


class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        # todo
        self.block1 = nn.Sequential(
            nn.Conv2d(inC, outC, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outC, outC, kernel_size=3, padding=1),
            nn.ReLU())

    def forward(self, x):
        # todo
        x = self.block1(x)
        return x


class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        # todo
        # Do not forget to concatenate with respective step in contracting path
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.withReLU = withReLU
        self.upconv = nn.ConvTranspose2d(inC, outC, kernel_size=2, stride = 2)
        self.conv1 = nn.Conv2d(inC, outC, kernel_size=3, padding =1)
        self.conv2 = nn.Conv2d(outC, outC, kernel_size=3, padding =1)



    def forward(self, x, x_down):
        # todo
        x = self.upconv(x)
        #print('x size' + str(x.size()))
        x = torch.cat((x, x_down), 1)
        #print('x concat  size' + str(x.size()))
        if self.withReLU:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
        else:
            x = self.conv1(x)
            x = self.conv2(x)

        return x


class maxPool(nn.Module):
    def __init__(self):
        super(maxPool, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class outputLayer(nn.Module):
    def __init__(self, inC, outC):
        super(outputLayer, self).__init__()
        # todo
        self.conv1 = nn.Sequential(nn.Conv2d(inC, outC, kernel_size=1),
                                   nn.Sigmoid()
                                   )

    def forward(self, x):
        # todo
        x = self.conv1(x)
        return x
