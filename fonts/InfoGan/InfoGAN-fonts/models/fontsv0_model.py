import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Architecture based on InfoGAN paper.
"""


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.tconv1 = nn.ConvTranspose2d(180, 2048, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2048)

        self.tconv2 = nn.ConvTranspose2d(2048, 1024, 4, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1024)

        self.tconv3 = nn.ConvTranspose2d(1024, 512, 4, 2, padding=1, bias=False)

        self.tconv4 = nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=False)

        self.tconv5 = nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False)
        self.tconv6 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
        self.tconv7 = nn.ConvTranspose2d(64, 3, 4, 2, padding=1, bias=False)


    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))

        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))
        x = F.relu(self.tconv5(x))
        x = F.relu(self.tconv6(x))


        img = torch.tanh(self.tconv7(x))

        return img


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, 4, 2, 1)

        self.conv2 = nn.Conv2d(8, 16, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)

        return x


class DHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(32, 1, 16)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))

        return output


class QHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(32, 512, 16, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 1024, 1)
        self.bn2 = nn.BatchNorm2d(1024)

        self.conv_disc = nn.Conv2d(1024, 100, 1)

        self.conv_mu = nn.Conv2d(1024, 2, 1)
        self.conv_var = nn.Conv2d(1024, 2, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn2(self.conv2(self.bn1(self.conv1(x)))), 0.1, inplace=True)

        disc_logits = self.conv_disc(x).squeeze()

        # Not used during training for celeba dataset.
        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var
