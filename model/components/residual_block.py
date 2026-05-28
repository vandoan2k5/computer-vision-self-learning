import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inc, outc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)

        self.shortcut = nn.Sequential()
        if(stride!=1 or inc != outc):
            self.shortcut = nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outc)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out