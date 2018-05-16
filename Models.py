import torch.nn as nn
import torch
from torchvision.models import resnet18, resnet

# class MyRestnet(resnet):
#     def __init__(self):
#         super(MyRestnet, self).__init__()
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x




class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = resnet18(pretrained=False)
        resnet.load_state_dict(torch.load('E:\PROJECT\FaceSRGAN\checkpoint\\resnet18-5c106cde.pth'))
        # Extracts features at the 11th layer
        ls = list(resnet.children())[:8]
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:7])

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
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 9, 1, 4),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [nn.Conv2d(64, 256, 3, 1, 1),
                           nn.BatchNorm2d(256),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, 9, 1, 4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


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


