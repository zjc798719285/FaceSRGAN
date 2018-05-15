import torch.nn as nn
import torch
from torchvision.models import vgg19



class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        vgg19_model = vgg19(pretrained=True)

        # Extracts features at the 11th layer
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:12])

    def forward(self, img):
        out = self.feature_extractor(img)
        return out



class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(in_features, in_features, 3, 1, 1)]
        self.conv_block = nn.Sequential(*conv_block)
    def forward(self, x):
        return x + self.conv_block(x)



class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=10):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 1, 3),
            nn.ReLU(inplace=True)
        )
        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)
        self.conv2 = nn.Sequential(nn.Conv2d(64, out_channels, 3, 1, 1))

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        return out2


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, 2, 3),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            ResidualBlock(64))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            ResidualBlock(128))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            ResidualBlock(256))
        self.linear = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        conv1 = self.conv1(img)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        output = torch.mean(torch.mean(conv4, -1), -1)
        output2 = self.linear(output)
        return output2


