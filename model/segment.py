import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
import cv2
import numpy as np

# from net.common import Conv, Focus, SEModule, autopad, fuse_conv_and_bn, init_head_s4


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class init_head_s4(nn.Module):
    def __init__(self, inplanes, planes, outplanes):
        super(init_head_s4, self).__init__()

        self.layer1 = Conv(inplanes, planes, k=5, s=2,
                           p=2, act=nn.PReLU(planes))
        self.layer2 = Conv(planes, outplanes - inplanes, k=5,
                           s=2, p=2, act=nn.PReLU(outplanes - inplanes))

    def forward(self, x):
        x_short = F.max_pool2d(x, kernel_size=4, stride=4)
        x_out = self.layer2(self.layer1(x))
        return torch.cat((x_short, x_out), dim=1)


class Conv(nn.Module):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=nn.Hardswish(), bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(
            k, p), groups=g, dilation=d, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = act if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


# fabby ver
class Bottleneck3x3(nn.Module):
    def __init__(self, inplanes, planes, pad=1, dilation=1):
        super(Bottleneck3x3, self).__init__()
        self.convs = nn.Sequential(
            # nn.Conv2d(inplanes, planes, kernel_size=1),
            # nn.BatchNorm2d(planes),
            # nn.PReLU(planes),
            Conv(inplanes, planes, k=1, act=nn.PReLU(planes)),

            # nn.Conv2d(planes, planes, kernel_size=3, padding=pad, dilation=dilation, groups=planes),
            # nn.BatchNorm2d(planes),
            # nn.PReLU(planes),
            Conv(planes, planes, k=3, p=pad, d=dilation,
                 g=planes, act=nn.PReLU(planes)),

            # nn.Conv2d(planes, inplanes, kernel_size=1),
            # nn.BatchNorm2d(inplanes),
            Conv(planes, inplanes, k=1, act=None),
        )
        self.prelu = nn.PReLU(inplanes)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        out += residual
        out = self.prelu(out)

        return out


class Bottleneck5x5(nn.Module):
    def __init__(self, inplanes, planes):
        super(Bottleneck5x5, self).__init__()
        self.convs = nn.Sequential(
            # nn.Conv2d(inplanes, planes, kernel_size=1),
            # nn.BatchNorm2d(planes),
            # nn.PReLU(planes),
            Conv(inplanes, planes, k=1, act=nn.PReLU(planes)),

            nn.Conv2d(planes, planes, kernel_size=(
                5, 1), padding=(2, 0), groups=planes),
            # nn.Conv2d(planes, planes, kernel_size=(1, 5), padding=(0, 2)),
            # nn.BatchNorm2d(planes),
            # nn.PReLU(planes),
            Conv(planes, planes, k=(1, 5), p=(0, 2),
                 g=planes, act=nn.PReLU(planes)),

            # nn.Conv2d(planes, inplanes, kernel_size=1),
            # nn.BatchNorm2d(inplanes),
            Conv(planes, inplanes, k=1, act=None),
        )
        self.prelu = nn.PReLU(inplanes)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        out += residual
        out = self.prelu(out)

        return out


class BottleneckDown2(nn.Module):
    def __init__(self, inplanes, planes, outplanes):
        super(BottleneckDown2, self).__init__()
        self.convs = nn.Sequential(
            # nn.Conv2d(inplanes, planes, kernel_size=2, stride=2),
            # nn.BatchNorm2d(planes),
            # nn.PReLU(planes),
            Conv(inplanes, planes, k=2, s=2, p=0,
                 act=nn.PReLU(planes)),  # ==!!!!==

            # nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, groups=planes),
            # nn.BatchNorm2d(planes),
            # nn.PReLU(planes),
            Conv(planes, planes, k=3, s=1, p=1,
                 g=planes, act=nn.PReLU(planes)),

            # nn.Conv2d(planes, outplanes, kernel_size=1),
            # nn.BatchNorm2d(outplanes),
            Conv(planes, outplanes, k=1, act=None),
        )
        self.convm = nn.Sequential(
            # nn.Conv2d(inplanes, outplanes, kernel_size=1),
            # nn.BatchNorm2d(outplanes)
            Conv(inplanes, outplanes, k=1, act=None),
        )

        self.prelu = nn.PReLU(outplanes)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        residual_1 = F.max_pool2d(residual, kernel_size=2, stride=2)
        residual = self.convm(residual_1)
        out += residual
        out = self.prelu(out)

        return out, residual_1


class BottleneckDim_Res(nn.Module):  # down dim
    def __init__(self, inplanes, planes, outplanes, usePrelu):
        super(BottleneckDim_Res, self).__init__()
        self.usePrelu = usePrelu
        if self.usePrelu:
            self.convs = nn.Sequential(
                # nn.Conv2d(inplanes, planes, kernel_size=1),
                # nn.BatchNorm2d(planes),
                # nn.PReLU(planes),
                Conv(inplanes, planes, k=1, act=nn.PReLU(planes)),

                # nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=planes),
                # nn.BatchNorm2d(planes),
                # nn.PReLU(planes),
                Conv(planes, planes, k=3, p=1, g=planes, act=nn.PReLU(planes)),

                # nn.Conv2d(planes, outplanes, kernel_size=1),
                # nn.BatchNorm2d(outplanes),
                Conv(planes, outplanes, k=1, act=None),
            )
        else:
            self.convs = nn.Sequential(
                # nn.Conv2d(inplanes, planes, kernel_size=1),
                # nn.BatchNorm2d(planes),
                # nn.ReLU(inplace=True),
                Conv(inplanes, planes, k=1, act=nn.PReLU(planes)),

                # nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=planes),
                # nn.BatchNorm2d(planes),
                # nn.ReLU(inplace=True),
                Conv(planes, planes, k=3, p=1, g=planes, act=nn.PReLU(planes)),

                # nn.Conv2d(planes, outplanes, kernel_size=1),
                # nn.BatchNorm2d(outplanes),
                Conv(planes, outplanes, k=1, act=None),
            )
        self.resconv = nn.Sequential(
            # nn.Conv2d(inplanes, outplanes, kernel_size=1),
            # nn.BatchNorm2d(outplanes)
            Conv(inplanes, outplanes, k=1, act=None),
        )

        self.prelu = nn.PReLU(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        residual = self.resconv(residual)
        out += residual

        if self.usePrelu:
            out = self.prelu(out)
        else:
            out = self.relu(out)

        return out


class BottleneckDim(nn.Module):  # down dim
    def __init__(self, inplanes, planes, outplanes, usePrelu):
        super(BottleneckDim, self).__init__()
        self.usePrelu = usePrelu
        if self.usePrelu:
            self.convs = nn.Sequential(
                # nn.Conv2d(inplanes, planes, kernel_size=1),
                # nn.BatchNorm2d(planes),
                # nn.PReLU(planes),
                Conv(inplanes, planes, k=1, act=nn.PReLU(planes)),

                # nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=planes),
                # nn.BatchNorm2d(planes),
                # nn.PReLU(planes),
                Conv(planes, planes, k=3, p=1, g=planes, act=nn.PReLU(planes)),

                # nn.Conv2d(planes, outplanes, kernel_size=1),
                # nn.BatchNorm2d(outplanes),
                Conv(planes, outplanes, k=1, act=None),
            )
        else:
            self.convs = nn.Sequential(
                # nn.Conv2d(inplanes, planes, kernel_size=1),
                # nn.BatchNorm2d(planes),
                # nn.ReLU(inplace=True),
                Conv(inplanes, planes, k=1, act=nn.ReLU(inplace=True)),

                # nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=planes),
                # nn.BatchNorm2d(planes),
                # nn.ReLU(inplace=True),
                Conv(planes, planes, k=3, p=1, act=nn.ReLU(inplace=True)),

                # nn.Conv2d(planes, outplanes, kernel_size=1),
                # nn.BatchNorm2d(outplanes),
                Conv(planes, outplanes, k=1, act=None),
            )

        self.prelu = nn.PReLU(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        out += residual
        if self.usePrelu:
            out = self.prelu(out)
        else:
            out = self.relu(out)

        return out


class BottleneckUp(nn.Module):  # upsample
    def __init__(self, inplanes, planes, outplanes):
        super(BottleneckUp, self).__init__()
        self.convs = nn.Sequential(
            # nn.Conv2d(inplanes, planes, kernel_size=1),
            # nn.BatchNorm2d(planes),
            # nn.ReLU(inplace=True),
            Conv(inplanes, planes, k=1, act=nn.ReLU(inplace=True)),

            nn.ConvTranspose2d(planes, planes, kernel_size=4,
                               padding=1, stride=2),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            # nn.Conv2d(planes, outplanes, kernel_size=1),
            # nn.BatchNorm2d(outplanes),
            Conv(planes, outplanes, k=1, act=None),
        )
        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        self.uppool = nn.MaxUnpool2d(2, stride=2)

    def forward(self, x, mp_indices):
        residual = x
        out = self.convs(x)
        residual = self.conv2(residual)
        residual = self.uppool(residual, mp_indices)
        out += residual
        out = F.relu(out, inplace=True)

        return out


class BottleneckUp_Res(nn.Module):  # upsample
    def __init__(self, inplanes, planes, outplanes):
        super(BottleneckUp_Res, self).__init__()
        self.convs = nn.Sequential(
            # nn.Conv2d(inplanes, planes, kernel_size=1),
            # nn.BatchNorm2d(planes),
            # nn.ReLU(inplace=True),
            Conv(inplanes, planes, k=1, act=nn.ReLU(inplace=True)),

            nn.ConvTranspose2d(planes, planes, kernel_size=4,
                               padding=1, stride=2),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            # nn.Conv2d(planes, outplanes, kernel_size=1),
            # nn.BatchNorm2d(outplanes),
            Conv(planes, outplanes, k=1, act=None),
        )

        self.conv2 = nn.Sequential(
            # nn.Conv2d(inplanes, outplanes, kernel_size=1),
            # nn.BatchNorm2d(outplanes)
            Conv(inplanes, outplanes, k=1, act=None),
        )

        self.uppool = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(outplanes * 2, outplanes, 1, 1, 0)
        )

    def forward(self, x, mp_indices):
        residual = x
        out = self.convs(x)
        residual = self.conv2(residual)
        # print(residual.size()[1], mp_indices.size()[1])
        residual = self.uppool(torch.cat([residual, mp_indices], 1))
        out += residual
        out = F.relu(out, inplace=True)

        return out


class BottleneckUp_Res_Other(BottleneckUp_Res):
    def __init__(self, inplanes, planes, outplanes, other):
        super().__init__(inplanes, planes, outplanes)
        self.uppool = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(outplanes + other, outplanes, 1, 1, 0)
        )


class Segment(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.export = False
        self.output_mid_features = False

        # # init section
        # self.init_Dim = 16
        # self.init_conv = Focus(3, self.init_Dim, k=3, s=1, act=nn.PReLU(self.init_Dim))

        # init section
        self.init_Dim = 16+in_channel
        self.init_conv = init_head_s4(in_channel, 16, self.init_Dim)

        # section 1
        self.bottle1_downDim = 16
        self.bottle1_Dim = 48
        self.bottle1_1 = BottleneckDown2(
            self.init_Dim, self.bottle1_downDim, self.bottle1_Dim)  # bottle 1_1
        self.bottle1_x = nn.Sequential(
            Bottleneck3x3(self.bottle1_Dim,
                          self.bottle1_downDim),  # bottle 1_2
            Bottleneck3x3(self.bottle1_Dim,
                          self.bottle1_downDim),  # bottle 1_3
            Bottleneck3x3(self.bottle1_Dim,
                          self.bottle1_downDim),  # bottle 1_4
            Bottleneck3x3(self.bottle1_Dim,
                          self.bottle1_downDim),  # bottle 1_5
        )

        # section 2
        self.bottle2_downDim = 48
        self.bottle2_Dim = 128
        self.bottle2_1 = BottleneckDown2(
            self.bottle1_Dim, self.bottle1_downDim, self.bottle2_Dim)
        self.bottle2_x = nn.Sequential(
            Bottleneck3x3(self.bottle2_Dim,
                          self.bottle2_downDim),  # bottle 2_2
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim,
                          pad=2, dilation=2),  # dilated 2  # bottle 2_3
            # Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim), #bottle 2_4
            # Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim, pad=2, dilation=2), # bottle 2_5
            Bottleneck3x3(self.bottle2_Dim,
                          self.bottle2_downDim),  # bottle 2_6
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim,
                          pad=4, dilation=4),  # dilated 4 # bottle 2_7
            # Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim),  # asymmetric 5  # bottle 2_8
            Bottleneck5x5(self.bottle2_Dim, self.bottle2_downDim),
            # SEModule(self.bottle2_Dim),
        )

        # section 3
        self.bottle3_1 = BottleneckDim_Res(
            self.bottle2_Dim * 2, self.bottle2_downDim, self.bottle2_Dim, usePrelu=True)
        # self.bottle3_1 = Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim)#bottle 3_1
        self.bottle3_x = nn.Sequential(
            Bottleneck3x3(self.bottle2_Dim,
                          self.bottle2_downDim),  # bottle 3_2
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim,
                          pad=2, dilation=2),  # dilated 2  # bottle 3_3
            Bottleneck3x3(self.bottle2_Dim,
                          self.bottle2_downDim),  # bottle 3_4
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim,
                          pad=4, dilation=4),  # bottle 2_5
            # Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim),  # bottle 3_6
            # Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim, pad=8, dilation=8),  # dilated 8  # bottle 3_7
            # Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim),  # bottle 3_8
            # Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim, pad=16, dilation=16),  # dilated 8  # bottle 3_8
            Bottleneck5x5(self.bottle2_Dim, self.bottle2_downDim),
            # SEModule(self.bottle2_Dim),
        )

        # section 4
        self.bottle4_1up = BottleneckUp_Res(
            self.bottle2_Dim, self.bottle1_downDim, self.bottle1_Dim)
        self.bottle4_2 = BottleneckDim_Res(
            self.bottle1_Dim * 2, 16, self.bottle1_Dim, usePrelu=False)
        #self.bottle4_2 = BottleneckDim(self.bottle1_Dim , 16, self.bottle1_Dim, usePrelu=False)
        self.bottle4_3 = BottleneckDim(
            self.bottle1_Dim, 16, self.bottle1_Dim, usePrelu=False)

        # section 5
        self.bottle5_1up = BottleneckUp_Res_Other(
            self.bottle1_Dim, 4, self.bottle1_downDim, self.init_Dim)
        self.bottle5_2 = BottleneckDim(
            self.bottle1_downDim, 4, self.bottle1_downDim, usePrelu=False)

        # section 6
        self.bottle6_1 = nn.ConvTranspose2d(
            self.bottle1_downDim,  4, kernel_size=8, padding=2, stride=4)
        self.bottle6_2 = nn.Conv2d(
            4, 1, kernel_size=3, padding=1)

        self.weights_init()

    # def weights_init(self):
    #     for idx, m in enumerate(self.modules()):
    #         classname = m.__class__.__name__
    #         if classname.find('Conv') != -1:
    #             m.weight.data.normal_(0.0, 0.02)
    #         elif classname.find('BatchNorm') != -1:
    #             m.weight.data.normal_(1.0, 0.02)
    #             m.bias.data.fill_

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # init section
        # init_out = self.init_conv(x)
        # init_mp = F.max_pool2d(x, kernel_size=2, stride=2)
        # init_down = torch.cat((init_out, init_mp), 1)

        init_down = self.init_conv(x)
        # pdb.set_trace()
        # init_down = self.init_bn(init_down)
        # init_down = self.init_prelu(init_down)

        # section 1
        bottle1_down, bottle1_indices = self.bottle1_1(init_down)  # bottle 1_1
        bottle1_5 = self.bottle1_x(bottle1_down)

        # section 2
        bottle2_down, bottle2_indices = self.bottle2_1(bottle1_5)  # bottle 2_1
        bottle2_8 = self.bottle2_x(bottle2_down)
        # concat_2
        concat_2 = torch.cat((bottle2_8, bottle2_down), 1)

        # section3
        bottle3_1 = self.bottle3_1(concat_2)
        bottle3_8 = self.bottle3_x(bottle3_1)

        # section4
        bottle4_1 = self.bottle4_1up(bottle3_8, bottle2_indices)
        # concat_1
        concat_1 = torch.cat((bottle1_down, bottle4_1), 1)

        bottle4_2 = self.bottle4_2(concat_1)
        bottle4_3 = self.bottle4_3(bottle4_2)

        # section5
        bottle5_1 = self.bottle5_1up(bottle4_3, bottle1_indices)
        bottle5_2 = self.bottle5_2(bottle5_1)

        # section6
        bottle6_1 = self.bottle6_1(bottle5_2)
        out = self.bottle6_2(bottle6_1)
        #out = bottle6_1

        return out

        if self.export == True:
            return torch.sigmoid(out)

        if self.output_mid_features == True:
            return out, [bottle1_5, bottle2_8, bottle3_8], [bottle3_8, bottle4_3, bottle5_2]

        if self.num_classes == 1:
            output3_1 = self.mid_fea3_1(bottle3_1)
            output4_1 = self.mid_fea4_1(bottle4_1)
            output5_1 = self.mid_fea5_1(bottle5_1)
            return out, [output3_1, output4_1, output5_1]
        elif self.num_classes == 2:
            # softmax_output = F.log_softmax(out,dim=1)
            softmax_output = F.softmax(out, dim=1)
            # section mid feas
            output3_1 = F.softmax(self.mid_fea3_1(bottle3_1), dim=1)
            output4_1 = F.softmax(self.mid_fea4_1(bottle4_1), dim=1)
            output5_1 = F.softmax(self.mid_fea5_1(bottle5_1), dim=1)
            # mask_out = Variable(softmax_output.data.max(1)[1])  # max(1)[0] are max values, max(1)[1] are idxs.
            return softmax_output[:, :1], [output5_1[:, :1], output4_1[:, :1], output3_1[:, :1]]

    def train_batch(self, x, heatmaps, pafs):
        inp = torch.cat([x, heatmaps,pafs], dim=1)
        out = self.forward(inp)
        return torch.sigmoid(out)


if __name__ == "__main__":
    from debug_function import *
    m = Segment(3+17)

    criterion = nn.BCELoss()

    x = torch.zeros((1, 3, 480, 480))
    heatmap = torch.zeros((1, 17, 480, 480))
    label = torch.ones((1, 1, 480, 480))

    o = m(x, heatmap)

    loss = criterion(o, label)
    loss.backward()

    # m = modshow(m, (17+3, 480, 480))

    check(o)
