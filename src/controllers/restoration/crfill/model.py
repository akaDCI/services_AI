import torch
import torch.nn as nn


class GenConv(nn.Conv2d):
    def __init__(self, cin, cout, ksize, stride=1, rate=1, activation=nn.ELU()):
        """Define conv for generator

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
            Stride: Convolution stride.
            rate: Rate for or dilated conv.
            activation: Activation function after convolution.
        """
        p = int(rate*(ksize-1)/2)
        super(GenConv, self).__init__(in_channels=cin, out_channels=cout,
                                      kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=1, bias=True)
        self.activation = activation

    def forward(self, x):
        x = super(GenConv, self).forward(x)
        if self.out_channels == 3 or self.activation is None:
            return x
        x, y = torch.split(x, int(self.out_channels/2), dim=1)
        x = self.activation(x)
        y = torch.sigmoid(y)
        x = x * y
        return x


class GenDeConv(GenConv):
    def __init__(self, cin, cout):
        """Define deconv for generator.
        The deconv is defined to be a x2 resize_nearest_neighbor operation with
        additional gen_conv operation.

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
        """
        super(GenDeConv, self).__init__(cin, cout, ksize=3)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2)
        x = super(GenDeConv, self).forward(x)
        return x


class BaseConvGenerator(nn.Module):
    def __init__(self, return_feat=False, return_pm=False):
        super(BaseConvGenerator, self).__init__()
        self.return_feat = return_feat
        self.return_pm = return_pm
        cnum = 48
        self.cnum = cnum
        # stage1
        self.conv1 = GenConv(5, cnum, 5, 1)
        self.conv2_downsample = GenConv(int(cnum/2), 2*cnum, 3, 2)
        self.conv3 = GenConv(cnum, 2*cnum, 3, 1)
        self.conv4_downsample = GenConv(cnum, 4*cnum, 3, 2)
        self.conv5 = GenConv(2*cnum, 4*cnum, 3, 1)
        self.conv6 = GenConv(2*cnum, 4*cnum, 3, 1)
        self.conv7_atrous = GenConv(2*cnum, 4*cnum, 3, rate=2)
        self.conv8_atrous = GenConv(2*cnum, 4*cnum, 3, rate=4)
        self.conv9_atrous = GenConv(2*cnum, 4*cnum, 3, rate=8)
        self.conv10_atrous = GenConv(2*cnum, 4*cnum, 3, rate=16)
        self.conv11 = GenConv(2*cnum, 4*cnum, 3, 1)
        self.conv12 = GenConv(2*cnum, 4*cnum, 3, 1)
        self.conv13_upsample_conv = GenDeConv(2*cnum, 2*cnum)
        self.conv14 = GenConv(cnum, 2*cnum, 3, 1)
        self.conv15_upsample_conv = GenDeConv(cnum, cnum)
        self.conv16 = GenConv(cnum//2, cnum//2, 3, 1)
        self.conv17 = GenConv(cnum//4, 3, 3, 1, activation=None)

        # stage2
        self.xconv1 = GenConv(3, cnum, 5, 1)
        self.xconv2_downsample = GenConv(cnum//2, cnum, 3, 2)
        self.xconv3 = GenConv(cnum//2, 2*cnum, 3, 1)
        self.xconv4_downsample = GenConv(cnum, 2*cnum, 3, 2)
        self.xconv5 = GenConv(cnum, 4*cnum, 3, 1)
        self.xconv6 = GenConv(2*cnum, 4*cnum, 3, 1)
        self.xconv7_atrous = GenConv(2*cnum, 4*cnum, 3, rate=2)
        self.xconv8_atrous = GenConv(2*cnum, 4*cnum, 3, rate=4)
        self.xconv9_atrous = GenConv(2*cnum, 4*cnum, 3, rate=8)
        self.xconv10_atrous = GenConv(2*cnum, 4*cnum, 3, rate=16)
        self.pmconv1 = GenConv(3, cnum, 5, 1)
        self.pmconv2_downsample = GenConv(cnum//2, cnum, 3, 2)
        self.pmconv3 = GenConv(cnum//2, 2*cnum, 3, 1)
        self.pmconv4_downsample = GenConv(cnum, 4*cnum, 3, 2)
        self.pmconv5 = GenConv(2*cnum, 4*cnum, 3, 1)
        self.pmconv6 = GenConv(2*cnum, 4*cnum, 3, 1,
                               activation=nn.ReLU())
        self.pmconv9 = GenConv(2*cnum, 4*cnum, 3, 1)
        self.pmconv10 = GenConv(2*cnum, 4*cnum, 3, 1)

        self.allconv11 = GenConv(4*cnum, 4*cnum, 3, 1)
        self.allconv12 = GenConv(2*cnum, 4*cnum, 3, 1)
        self.allconv13_upsample_conv = GenDeConv(2*cnum, 2*cnum)
        self.allconv14 = GenConv(cnum, 2*cnum, 3, 1)
        self.allconv15_upsample_conv = GenDeConv(cnum, cnum)
        self.allconv16 = GenConv(cnum//2, cnum//2, 3, 1)
        self.allconv17 = GenConv(cnum//4, 3, 3, 1, activation=None)

    def forward(self, image, mask):
        # Preprocessing inputs
        mask = (mask > 0) * 1
        inputs = image * (1 - mask)
        x = inputs

        xin = x
        bsize, ch, height, width = x.shape
        ones_x = torch.ones(bsize, 1, height, width).to(x.device)
        x = torch.cat([x, ones_x, ones_x*mask], 1)

        # two stage network
        # stage1
        x = self.conv1(x)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13_upsample_conv(x)
        x = self.conv14(x)
        x = self.conv15_upsample_conv(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = torch.tanh(x)
        x_stage1 = x

        x = x*mask + xin[:, 0:3, :, :]*(1.-mask)
        xnow = x

        ###
        x = self.xconv1(xnow)
        x = self.xconv2_downsample(x)
        x = self.xconv3(x)
        x = self.xconv4_downsample(x)
        x = self.xconv5(x)
        x = self.xconv6(x)
        x = self.xconv7_atrous(x)
        x = self.xconv8_atrous(x)
        x = self.xconv9_atrous(x)
        x = self.xconv10_atrous(x)
        x_hallu = x

        ###
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        pm_return = x

        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], 1)

        x = self.allconv11(x)
        x = self.allconv12(x)
        x = self.allconv13_upsample_conv(x)
        x = self.allconv14(x)
        x = self.allconv15_upsample_conv(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.tanh(x)
        if self.return_pm:
            return x_stage1, x_stage2, pm_return

        # Posprocessing image
        x_stage2 = x_stage2 * mask + inputs * (1 - mask)
        x_stage2 = torch.clamp(x_stage2, -1, 1) * 255.

        return x_stage2
