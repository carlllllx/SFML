from torch import nn
from torch.nn import functional as F


class resnetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(resnetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class resnetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(resnetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        extra_x = self.conv3(x)
        extra_x = self.bn3(extra_x)
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(extra_x + output)


class resnet18(nn.Module):
    def __init__(self, args):
        super(resnet18, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(resnetBasicBlock(64, 64, 1),
                                    resnetBasicBlock(64, 64, 1))
        self.layer2 = nn.Sequential(resnetDownBlock(64, 128, [2, 1]),
                                    resnetBasicBlock(128, 128, 1))
        self.layer3 = nn.Sequential(resnetDownBlock(128, 256, [2, 1]),
                                    resnetBasicBlock(256, 256, 1))
        self.layer4 = nn.Sequential(resnetDownBlock(256, 512, [2, 1]),
                                    resnetBasicBlock(512, 512, 1))
        self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, self.args.num_classes)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avgpool1(output)
        output = output.reshape(x.shape[0], -1)
        output = self.fc(output)
        return output


class resnet18_client_side_part1(nn.Module):
    def __init__(self, args):
        super(resnet18_client_side_part1, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        return output


class resnet18_server_side_middle(nn.Module):
    def __init__(self, args):
        super(resnet18_server_side_middle, self).__init__()
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(resnetBasicBlock(64, 64, 1),
                                    resnetBasicBlock(64, 64, 1))
        self.layer2 = nn.Sequential(resnetDownBlock(64, 128, [2, 1]),
                                    resnetBasicBlock(128, 128, 1))
        self.layer3 = nn.Sequential(resnetDownBlock(128, 256, [2, 1]),
                                    resnetBasicBlock(256, 256, 1))
        self.layer4 = nn.Sequential(resnetDownBlock(256, 512, [2, 1]),
                                    resnetBasicBlock(512, 512, 1))
        self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        output = self.relu1(x)
        output = self.maxpool1(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avgpool1(output)
        output = output.reshape(x.shape[0], -1)
        return output


class resnet18_client_side_part2(nn.Module):
    def __init__(self, args):
        super(resnet18_client_side_part2, self).__init__()
        self.args = args
        self.fc = nn.Linear(512, self.args.num_classes)

    def forward(self, x):
        output = self.fc(x)
        return output