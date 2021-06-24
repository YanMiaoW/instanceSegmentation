import torch
import torch.nn as nn
import torch.nn.functional as F


class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def name():
        print("aa")


def mish(x):
    return x * torch.tanh(F.softplus(x))


def CBL(i, o):
    return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.LeakyReLU(0.1)]


def CBM(i, o):
    return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), mish(o)]


def CBL1x1(i, o):
    return [nn.Conv2d(i, o, 1, padding=0), nn.BatchNorm2d(o), nn.LeakyReLU(0.1)]


def CBM1x1(i, o):
    return [nn.Conv2d(i, o, 1, padding=0), nn.BatchNorm2d(o), mish(o)]


def CBLDown(i, o):
    return [
        nn.Conv2d(i, o, 3, stride=2, padding=1),
        nn.BatchNorm2d(o),
        nn.LeakyReLU(0.1),
    ]


def CBMDown(i, o):
    return [nn.Conv2d(i, o, 3, stride=2, padding=1), nn.BatchNorm2d(o), mish(o)]


class ResBlock(Module):
    def __init__(self, i):
        self.net = [CBM1x1(i, i // 2), CBM(i // 2, i)]
        super().__init__()

    def forward(self, x):
        return x + self.net(x)


class CSP(Module):
    def __init__(self, i, n):
        o = i * 2

        self.down = CBMDown(i, o)

        self.dense = [CBM1x1(o, o)]
        self.dense += [ResBlock(o)] * n
        self.dense += [CBM1x1(o, o)]

        self.fast = [CBM1x1(o, o)]

        self.out = CBM1x1(o * 2, o)

        super().__init__()

    def forward(self, x):
        x = self.down(x)

        y1 = self.dense(x)
        y2 = self.fast(x)

        z = torch.cat((y1, y2), dim=0)

        return self.out(z)


class CSPDarknet53(Module):
    def __init__(self):

        self.init = [CBM(3, 32), CBMDown(32, 64)]

        self.stage3 = [CSP(64, 1)]
        self.stage3 += [CSP(128, 2)]
        self.stage3 += [CSP(256, 8)]

        self.stage2 = [CSP(512, 8)]

        self.stage1 = [CSP(1024, 4)]

        super().__init__()

    def forward(self, f1):
        f2 = self.init(f1)
        f8 = self.stage3(f2)
        f16 = self.stage2(f8)
        f32 = self.stage1(f16)

        return f8, f16, f32


def spp(x):
    pool5x5 = torch.max_pool2d(x, 5, stride=1, padding=2)
    pool9x9 = torch.max_pool2d(x, 9, stride=1, padding=4)
    pool13x13 = torch.max_pool2d(x, 13, stride=1, padding=6)
    return torch.cat((x, pool5x5, pool9x9, pool13x13), dim=0)


class PAN(Module):
    def __init__(self):
        self.up1 = [CBL1x1(512, 256)]
        self.up1 += [nn.UpsamplingBilinear2d(scale_factor=2)]

        self.s1 = CBL1x1(1024, 256)

        self.up2 = [
            CBL1x1(512, 256),
            CBL(256, 512),
            CBL1x1(512, 256),
            CBL(256, 512),
            CBL1x1(512, 256),
            CBL1x1(256, 128),
        ]
        self.up2 += [nn.UpsamplingBilinear2d(scale_factor=2)]

        self.s2 = CBL1x1(512, 128)

        super().__init__()

    def forward(self, x, f16, f32):
        y1 = self.up1(x)
        y2 = self.s1(f16)

        o1 = torch.cat((y1, y2), dim=0)

        y1 = self.up2(o1)
        y2 = self.s2(f32)

        o2 = torch.cat((y1, y2), dim=0)
        
        

        pass


class Yolov4(Module):
    def __init__(self):
        super().__init__()
        self.backbone = CSPDarknet53()

        self.spp = [CBL1x1(1024, 512), CBL(512, 1024), CBL1x1(1024, 512)]
        self.spp += [spp]
        self.spp += [CBL1x1(2048, 512), CBL(512, 1024), CBL1x1(1024, 512)]

        self.pan = [PAN()]

    def forward(self, x):
        s8, s16, s32 = self.backbone(x)
        
        o = self.spp(s32)


if __name__ == "__main__":
    m = CSPDarknet53()
    i = torch.zeros((1, 3, 608, 608))
    o = m(i)
    print(o.shape)
