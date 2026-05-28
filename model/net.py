import torch
import torch.nn as nn
from .components import ResidualBlock

class CIFAR100Net(nn.Module):
    def __init__(self, num_classes = 100):
        super(CIFAR100Net, self).__init__()
        self.inc = 64
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def _make_layer(self, outc, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.inc, outc, s))
            self.inc = outc
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.prep(x)     # Kích thước: [Batch, 64, 32, 32]
        
        x = self.layer1(x)   # Kích thước: [Batch, 64, 32, 32]
        x = self.layer2(x)   # Kích thước: [Batch, 128, 16, 16]
        x = self.layer3(x)   # Kích thước: [Batch, 256, 8, 8]
        x = self.layer4(x)   # Kích thước: [Batch, 512, 4, 4]
        
        x = self.avgpool(x)  # Kích thước: [Batch, 512, 1, 1]
        x = torch.flatten(x, 1) # Kích thước: [Batch, 512]
        x = self.fc(x)       # Kích thước: [Batch, 100]
        return x