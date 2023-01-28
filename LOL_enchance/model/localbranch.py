import torch.nn as nn
import math

from timm.models.layers import trunc_normal_

from model.wsab import WsabBlock

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ResidualModule(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(ResidualModule, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 
    
class ResidualTransformerModule(nn.Module):
    def __init__(self, inp_dim, out_dim, num_heads=4, window_size=4):
        super(ResidualTransformerModule, self).__init__()
        self.wsab1 = WsabBlock(dim=inp_dim, num_heads=num_heads, window_size=window_size)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.wsab1(x)
        # out = self.conv(out)
        out += residual
        return out 

class LocalEnhancementBlock(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(LocalEnhancementBlock, self).__init__()
        nf = f + increase
        self.up1 = ResidualModule(f, f)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = ResidualTransformerModule(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = LocalEnhancementBlock(n-1, nf, bn=bn)
        else:
            self.low2 = ResidualTransformerModule(nf, nf)
        self.low3 = ResidualTransformerModule(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2

class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = ResidualModule(f, f)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = ResidualModule(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn)
        else:
            self.low2 = ResidualModule(nf, nf)
        self.low3 = ResidualModule(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2

class LocalNet(nn.Module):
    def __init__(self, in_dim=3, dim=16):
        super(LocalNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.mul_blocks = nn.Sequential(LocalEnhancementBlock(n=4, f=16),
                                        LocalEnhancementBlock(n=4, f=16),
                                        LocalEnhancementBlock(n=4, f=16))
        self.add_blocks = nn.Sequential(LocalEnhancementBlock(n=4, f=16),
                                        LocalEnhancementBlock(n=4, f=16),
                                        LocalEnhancementBlock(n=4, f=16))

        # self.mul_blocks = nn.Sequential(Hourglass(n=4, f=16),
        #                                 Hourglass(n=4, f=16),
        #                                 Hourglass(n=4, f=16))
        # self.add_blocks = nn.Sequential(Hourglass(n=4, f=16),
        #                                 Hourglass(n=4, f=16),
        #                                 Hourglass(n=4, f=16))
                
        self.mul_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        # short cut connection
        mul = self.mul_blocks(x) + x
        add = self.add_blocks(x) + x
        mul = self.mul_end(mul)
        add = self.add_end(add)
        return mul, add
