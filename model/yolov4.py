import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction

from debug_function import *


class Module(nn.Module):
    def __init__(self, *l, **d):
        super().__init__()

        self.init(*l, **d)

        # [nn.module, ...] -> nn.Sequential(nn.module, ...)
        for key, value in self.__dict__.copy().items():
            if isinstance(value, list) and "_" not in key:
                seq = self._list2seq(value)
                setattr(self, key, seq)

    def _list2seq(self, s0):

        s = []

        for i in s0:

            if isinstance(i, nn.Module):
                s.append(i)

            elif isinstance(i, list):
                # 递归 flatten list
                s.append(self._list2seq(i))

            elif isfunction(i):
                # 函数 包装成 nn.module对象
                _funci = i  # 用i会和全局的i冲突
                M = type(i.__name__,(nn.Module,),{"forward": lambda self, x: _funci(x)})
                s.append(M())

            else:
                assert False, "not support " + str(type(i))

        return nn.Sequential(*s)


def mish(x):
    return x * torch.tanh(F.softplus(x))


def CBL(i, o):
    return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.LeakyReLU(0.1)]


def CBM(i, o):
    return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), mish]


def CBL1x1(i, o):
    return [nn.Conv2d(i, o, 1, padding=0), nn.BatchNorm2d(o), nn.LeakyReLU(0.1)]


def CBM1x1(i, o):
    return [nn.Conv2d(i, o, 1, padding=0), nn.BatchNorm2d(o), mish]


def CBLDown(i, o):
    return [
        nn.Conv2d(i, o, 3, stride=2, padding=1),
        nn.BatchNorm2d(o),
        nn.LeakyReLU(0.1),
    ]


def CBMDown(i, o):
    return [nn.Conv2d(i, o, 3, stride=2, padding=1), nn.BatchNorm2d(o), mish]


class ResBlock(Module):
    def init(self, i):
        self.net = [CBM1x1(i, i // 2), CBM(i // 2, i)]

    def forward(self, x):
        return x + self.net(x)


class CSP(Module):
    def init(self, i, n):
        o = i * 2

        self.down = CBMDown(i, o)

        self.dense = [CBM1x1(o, o)]
        self.dense += [ResBlock(o) for i in range(n)]
        self.dense += [CBM1x1(o, o)]

        self.fast = [CBM1x1(o, o)]

        self.out = CBM1x1(o * 2, o)

    def forward(self, x):
        x = self.down(x)

        y1 = self.dense(x)
        y2 = self.fast(x)

        z = torch.cat((y1, y2), dim=1)

        return self.out(z)


class CSPDarknet53(Module):
    def init(self):
        self.init1 = [CBM(3, 32)]

        self.stage3 = [CSP(32, 1)]
        self.stage3 += [CSP(64, 2)]
        self.stage3 += [CSP(128, 8)]

        self.stage2 = [CSP(256, 8)]

        self.stage1 = [CSP(512, 4)]

    def forward(self, f1):
        f2 = self.init1(f1)
        f8 = self.stage3(f2)
        f16 = self.stage2(f8)
        f32 = self.stage1(f16)

        return f8, f16, f32


def spp(x):
    pool5x5 = F.max_pool2d(x, 5, stride=1, padding=2)
    pool9x9 = F.max_pool2d(x, 9, stride=1, padding=4)
    pool13x13 = F.max_pool2d(x, 13, stride=1, padding=6)
    return torch.cat((x, pool5x5, pool9x9, pool13x13), dim=1)


def Cx5(i):
    # i -> i // 2
    o = i // 2
    return [
        CBL1x1(i, o),
        CBL(o, i),
        CBL1x1(i, o),
        CBL(o, i),
        CBL1x1(i, o),
    ]


class FPN(Module):
    def init(self):
        self.up1 = [CBL1x1(512, 256)]
        self.up1 += [nn.UpsamplingBilinear2d(scale_factor=2)]

        self.s1 = CBL1x1(512, 256)
        # cat 256 -> 512
        self.cx1 = Cx5(512)

        self.up2 = [CBL1x1(256, 128)]
        self.up2 += [nn.UpsamplingBilinear2d(scale_factor=2)]

        self.s2 = CBL1x1(256, 128)
        # cat 128 -> 256
        self.cx2 = Cx5(256)

    def forward(self, f8, f16, f32):
        up16 = self.up1(f32)
        s16 = self.s1(f16)

        o16 = torch.cat((up16, s16), dim=1)
        cx16 = self.cx1(o16)

        up8 = self.up2(cx16)
        s8 = self.s2(f8)

        o8 = torch.cat((up8, s8), dim=1)
        cx8 = self.cx2(o8)

        return cx8, cx16


class PA(Module):
    def init(self):
        self.down1 = [CBLDown(128, 256)]
        # cat 256 -> 512
        self.cx1 = Cx5(512)

        self.down2 = [CBLDown(256, 512)]
        # cat 512 -> 1024
        self.cx2 = Cx5(1024)

    def forward(self, f8, f16, f32):
        d16 = self.down1(f8)

        o16 = torch.cat((f16, d16), dim=1)
        cx16 = self.cx1(o16)

        d32 = self.down2(cx16)

        o32 = torch.cat((f32, d32), dim=1)
        cx32 = self.cx2(o32)

        return cx16, cx32


class Head(Module):
    def init(self, classes):
        o = 3 * (4 + 1 + classes)

        self.o1 = [CBL(128, 256)]
        self.o1 += [CBL1x1(256, o)]

        self.o2 = [CBL(256, 512)]
        self.o2 += [CBL1x1(512, o)]

        self.o3 = [CBL(512, 1024)]
        self.o3 += [CBL1x1(1024, o)]

    def forward(self, f8, f16, f32):
        o8 = self.o1(f8)
        o16 = self.o2(f16)
        o32 = self.o3(f32)

        return o8, o16, o32


class Yolov4(Module):
    def init(self, classes=80):
        self.backbone = CSPDarknet53()

        self.spp = [CBL1x1(1024, 512), CBL(512, 1024), CBL1x1(1024, 512)]
        self.spp += [spp]
        self.spp += [CBL1x1(2048, 512), CBL(512, 1024), CBL1x1(1024, 512)]

        self.fpn = FPN()

        self.pa = PA()

        self.h = Head(classes)

    def forward(self, x):
        f8, f16, f32 = self.backbone(x)
        s32 = self.spp(f32)

        p8, p16 = self.fpn(f8, f16, s32)

        n16, n32 = self.pa(p8, p16, s32)

        o8, o16, o32 = self.h(p8, n16, n32)

        return o8, o16, o32


if __name__ == "__main__":
    # m = CSPDarknet53()
    
    m = Yolov4()
    # i = torch.zeros((1, 3, 608, 608))
    # o = F.max_pool2d(i, 5, stride=1, padding=2)
    modshow(m,(1,3,608,608))
    
    # a, b, c = m(i)
    # print(a.shape, b.shape, c.shape)
