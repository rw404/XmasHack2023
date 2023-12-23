import numpy as np
import torch
import triton_python_backend_utils as pb_utils

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision.transforms.functional import to_tensor


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU,
                 batch_norm=False):
        super(ConvBlock, self).__init__()
        conv_type = OPT['conv_type']
        if conv_type == 'conv':
            self.conv = nn.Conv2d(int(inc), int(
                outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        elif conv_type.startswith('drconv'):
            region_num = int(conv_type.replace('drconv', ''))
            self.conv = DRConv2d(int(inc), int(outc), kernel_size, region_num=region_num, padding=padding,
                                 stride=stride)
            print(f'[ WARN ] Using DRconv2d(n_region={region_num}) instead of Conv2d in BilateralUpsampleNet.')
        else:
            raise NotImplementedError()

        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class FC(nn.Module):
    def __init__(self, inc, outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc), bias=(not batch_norm))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class SliceNode(nn.Module):
    def __init__(self, opt):
        super(SliceNode, self).__init__()
        self.opt = opt

    def forward(self, bilateral_grid, guidemap):
        # bilateral_grid shape: Nx12x8x16x16
        device = bilateral_grid.get_device()
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid(
            [torch.arange(0, H), torch.arange(0, W)])  # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)

        hg = hg.float().repeat(N, 1, 1).unsqueeze(
            3) / (H - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(
            3) / (W - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
        guidemap = guidemap * 2 - 1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)

        # guidemap shape: [N, 1 (D), H, W]
        # bilateral_grid shape: [N, 12 (c), 8 (d), 16 (h), 16 (w)], which is considered as a 3D space: [8, 16, 16]
        # guidemap_guide shape: [N, 1 (D), H, W, 3], which is considered as a 3D space: [1, H, W]
        # coeff shape: [N, 12 (c), 1 (D), H, W]

        # in F.grid_sample, gird is guidemap_guide, input is bilateral_grid
        # guidemap_guide[N, D, H, W] is a 3-vector <x, y, z>. but:
        #       x -> W, y -> H, z -> D  in bilater_grid
        # What does it really do:
        #   [ 1 ] For pixel in guidemap_guide[D, H, W], get <x,y,z>, and:
        #   [ 2 ] Normalize <x, y, z> from [-1, 1] to [0, w - 1], [0, h - 1], [0, d - 1], respectively.
        #   [ 3 ] Locate pixel in bilateral_grid at position [N, :, z, y, x].
        #   [ 4 ] Interplate using the neighbor values as the output affine matrix.

        # Force them have the same type for fp16 training :
        guidemap_guide = guidemap_guide.type_as(bilateral_grid)
        # bilateral_grid = bilateral_grid.type_as(guidemap_guide)
        coeff = F.grid_sample(bilateral_grid, guidemap_guide,
                              'bilinear', align_corners=True)
        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    def forward(self, coeff, full_res_input):
        '''
        coeff shape: [bs, 12, h, w]
        input shape: [bs, 3, h, w]
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''
        R = torch.sum(
            full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(
            full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(
            full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)


class ApplyCoeffsGamma(nn.Module):
    def __init__(self):
        super(ApplyCoeffsGamma, self).__init__()
        print('[ WARN ] Use alter methods indtead of affine matrix.')

    def forward(self, x_r, x):
        '''
        coeff shape: [bs, 12, h, w]
        apply zeroDCE curve.
        '''

        # [ 008 ] single iteration alpha map:
        # coeff channel num: 3
        # return x + x_r * (torch.pow(x, 2) - x)

        # [ 009 ] 8 iteratoins:
        # coeff channel num: 24
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * \
            (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image

        # [ 014 ] use illu map:
        # coeff channel num: 3
        # return x / (torch.where(x_r < x, x, x_r) + 1e-7)

        # [ 015 ] use HSV and only affine V channel:
        # coeff channel num: 3
        # V = torch.sum(x * x_r, dim=1, keepdim=True) + x_r
        # return torch.cat([x[:, 0:2, ...], V], dim=1)


class ApplyCoeffsRetinex(nn.Module):
    def __init__(self):
        super().__init__()
        print('[ WARN ] Use alter methods indtead of affine matrix.')

    def forward(self, x_r, x):
        '''
        coeff shape: [bs, 12, h, w]
        apply division of illumap.
        '''

        # [ 014 ] use illu map:
        # coeff channel num: 3
        return x / (torch.where(x_r < x, x, x_r) + 1e-7)


class GuideNet(nn.Module):
    def __init__(self, params=None, out_channel=1):
        super(GuideNet, self).__init__()
        self.params = params
        self.conv1 = ConvBlock(3, 16, kernel_size=1,
                               padding=0, batch_norm=True)
        self.conv2 = ConvBlock(
            16, out_channel, kernel_size=1, padding=0, activation=nn.Sigmoid)  # nn.Tanh

    def forward(self, x):
        return self.conv2(self.conv1(x))  # .squeeze(1)


class LowResNet(nn.Module):
    def __init__(self, coeff_dim=12, opt=None):
        super(LowResNet, self).__init__()
        self.params = opt
        self.coeff_dim = coeff_dim

        lb = opt[LUMA_BINS]
        cm = opt[CHANNEL_MULTIPLIER]
        sb = opt[SPATIAL_BIN]
        bn = opt[BATCH_NORM]
        nsize = opt[LOW_RESOLUTION]

        self.relu = nn.ReLU()

        # splat features
        n_layers_splat = int(np.log2(nsize / sb))
        self.splat_features = nn.ModuleList()
        prev_ch = 3
        for i in range(n_layers_splat):
            use_bn = bn if i > 0 else False
            self.splat_features.append(
                ConvBlock(prev_ch, cm * (2 ** i) * lb, 3, stride=2, batch_norm=use_bn))
            prev_ch = splat_ch = cm * (2 ** i) * lb

        # global features
        n_layers_global = int(np.log2(sb / 4))
        # print(n_layers_global)
        self.global_features_conv = nn.ModuleList()
        self.global_features_fc = nn.ModuleList()
        for i in range(n_layers_global):
            self.global_features_conv.append(
                ConvBlock(prev_ch, cm * 8 * lb, 3, stride=2, batch_norm=bn))
            prev_ch = cm * 8 * lb

        n_total = n_layers_splat + n_layers_global
        prev_ch = prev_ch * (nsize / 2 ** n_total) ** 2
        self.global_features_fc.append(
            FC(prev_ch, 32 * cm * lb, batch_norm=bn))
        self.global_features_fc.append(
            FC(32 * cm * lb, 16 * cm * lb, batch_norm=bn))
        self.global_features_fc.append(
            FC(16 * cm * lb, 8 * cm * lb, activation=None, batch_norm=bn))

        # local features
        self.local_features = nn.ModuleList()
        self.local_features.append(
            ConvBlock(splat_ch, 8 * cm * lb, 3, batch_norm=bn))
        self.local_features.append(
            ConvBlock(8 * cm * lb, 8 * cm * lb, 3, activation=None, use_bias=False))

        # predicton
        self.conv_out = ConvBlock(
            8 * cm * lb, lb * coeff_dim, 1, padding=0, activation=None)

    def forward(self, lowres_input):
        params = self.params
        bs = lowres_input.shape[0]
        lb = params[LUMA_BINS]
        cm = params[CHANNEL_MULTIPLIER]
        sb = params[SPATIAL_BIN]

        x = lowres_input
        for layer in self.splat_features:
            x = layer(x)
        splat_features = x

        for layer in self.global_features_conv:
            x = layer(x)
        x = x.view(bs, -1)
        for layer in self.global_features_fc:
            x = layer(x)
        global_features = x

        x = splat_features
        for layer in self.local_features:
            x = layer(x)
        local_features = x

        # shape: bs x 64 x 16 x 16
        fusion_grid = local_features

        # shape: bs x 64 x 1 x 1
        fusion_global = global_features.view(bs, 8 * cm * lb, 1, 1)
        fusion = self.relu(fusion_grid + fusion_global)

        x = self.conv_out(fusion)

        # reshape channel dimension -> bilateral grid dimensions:
        # [bs, 96, 16, 16] -> [bs, 12, 8, 16, 16]
        y = torch.stack(torch.split(x, self.coeff_dim, 1), 2)
        return y


class DRDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, **kargs):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            DRConv2d(in_channels, mid_channels, kernel_size=3,
                     region_num=REGION_NUM_, padding=1, **kargs),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DRConv2d(mid_channels, out_channels, kernel_size=3,
                     region_num=REGION_NUM_, padding=1, **kargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        assert len(DRCONV_POSITION_) == 2
        assert DRCONV_POSITION_[0] or DRCONV_POSITION_[1]
        if DRCONV_POSITION_[0] == 0:
            print('[ WARN ] Use Conv in DRDoubleConv[0] instead of DRconv.')
            self.double_conv[0] = nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding=1)
        if DRCONV_POSITION_[1] == 0:
            print('[ WARN ] Use Conv in DRDoubleConv[3] instead of DRconv.')
            self.double_conv[3] = nn.Conv2d(
                mid_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        res = self.double_conv(x)
        self.guide_features = []
        if DRCONV_POSITION_[0]:
            self.guide_features.append(self.double_conv[0].guide_feature)
        if DRCONV_POSITION_[1]:
            self.guide_features.append(self.double_conv[3].guide_feature)
        return res


class HistDRDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = HistDRConv2d(
            in_channels, mid_channels, kernel_size=3, region_num=REGION_NUM_, padding=1)
        self.inter1 = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = HistDRConv2d(
            mid_channels, out_channels, kernel_size=3, region_num=REGION_NUM_, padding=1)
        self.inter2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, histmap):
        y = self.conv1(x, histmap)
        y = self.inter1(y)
        y = self.conv2(y, histmap)
        return self.inter2(y)


class HistGuidedDRDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, **kargs):
        super().__init__()
        assert len(DRCONV_POSITION_) == 2
        assert DRCONV_POSITION_[0] or DRCONV_POSITION_[1]

        if not mid_channels:
            mid_channels = out_channels
        if DRCONV_POSITION_[0]:
            self.conv1 = DRConv2d(
                in_channels, mid_channels, kernel_size=3, region_num=REGION_NUM_, padding=1, **kargs)
        else:
            print('[ WARN ] Use Conv in HistGuidedDRDoubleConv[0] instead of DRconv.')
            self.conv1 = nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding=1)

        self.inter1 = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        if DRCONV_POSITION_[1]:
            self.conv2 = DRConv2d(
                mid_channels, out_channels, kernel_size=3, region_num=REGION_NUM_, padding=1, **kargs)
        else:
            print('[ WARN ] Use Conv in HistGuidedDRDoubleConv[0] instead of DRconv.')
            self.conv2 = nn.Conv2d(
                mid_channels, out_channels, kernel_size=3, padding=1)

        self.inter2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, histmap):
        if DRCONV_POSITION_[0]:
            y = self.conv1(x, histmap)
        else:
            y = self.conv1(x)
        y = self.inter1(y)

        if DRCONV_POSITION_[1]:
            y = self.conv2(y, histmap)
        else:
            y = self.conv2(y)

        # self.guide_features = [self.conv1.guide_feature, self.conv2.guide_feature]
        self.guide_features = []
        if DRCONV_POSITION_[0]:
            self.guide_features.append(self.conv1.guide_feature)
        if DRCONV_POSITION_[1]:
            self.guide_features.append(self.conv2.guide_feature)

        return self.inter2(y)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, **kargs):
        super().__init__()
        self.up = nn.Upsample(scale_factor=DOWN_RATIO_,
                              mode='bilinear', align_corners=True)
        if CONV_TYPE_ == 'drconv':
            if HIST_AS_GUIDE_:
                self.conv = HistDRDoubleConv(
                    in_channels, out_channels, in_channels // 2)
            elif GUIDE_FEATURE_FROM_HIST_:
                self.conv = HistGuidedDRDoubleConv(
                    in_channels, out_channels, in_channels // 2, **kargs)
            else:
                self.conv = DRDoubleConv(
                    in_channels, out_channels, in_channels // 2)
        # elif CONV_TYPE_ == 'dconv':
        #     self.conv = HistDyDoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2, histmap):
        """
        histmap: shape [bs, c * n_bins, h, w]
        """
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        if HIST_AS_GUIDE_ or GUIDE_FEATURE_FROM_HIST_ or CONV_TYPE_ == 'dconv':
            x = torch.cat([x2, x1], dim=1)
            res = self.conv(x, histmap)
        else:
            x = torch.cat([x2, x1, histmap], dim=1)
            res = self.conv(x)
        self.guide_features = self.conv.guide_features
        return res


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_hist=False):
        super().__init__()
        self.use_hist = use_hist
        if not use_hist:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(DOWN_RATIO_),
                DoubleConv(in_channels, out_channels)
            )
        else:
            if HIST_AS_GUIDE_:
                # self.maxpool_conv = nn.Sequential(
                #     nn.MaxPool2d(2),
                #     HistDRDoubleConv(in_channels, out_channels, in_channels // 2)
                # )
                raise NotImplementedError()
            elif GUIDE_FEATURE_FROM_HIST_:
                self.maxpool = nn.MaxPool2d(DOWN_RATIO_)
                self.conv = HistGuidedDRDoubleConv(
                    in_channels, out_channels, in_channels // 2)
            else:
                self.maxpool_conv = nn.Sequential(
                    nn.MaxPool2d(DOWN_RATIO_),
                    DRDoubleConv(in_channels, out_channels, in_channels // 2)
                )

    def forward(self, x, histmap=None):
        if GUIDE_FEATURE_FROM_HIST_ and self.use_hist:
            x = self.maxpool(x)
            return self.conv(x, histmap)
        elif self.use_hist:
            return self.maxpool_conv(torch.cat([x, histmap], axis=1))
        else:
            return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class HistUNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 bilinear=True,
                 n_bins=8,
                 hist_as_guide=False,
                 channel_nums=None,
                 hist_conv_trainable=False,
                 encoder_use_hist=False,
                 guide_feature_from_hist=False,
                 region_num=8,
                 use_gray_hist=False,
                 conv_type='drconv',
                 down_ratio=1,
                 drconv_position=[1, 1],
                 ):
        super().__init__()
        C_NUMS = [16, 32, 64, 128, 256]
        if channel_nums:
            C_NUMS = channel_nums
        self.maxpool = nn.MaxPool2d(2)
        self.n_bins = n_bins
        self.encoder_use_hist = encoder_use_hist
        self.use_gray_hist = use_gray_hist
        self.hist_conv_trainable = hist_conv_trainable

        global HIST_AS_GUIDE_, GUIDE_FEATURE_FROM_HIST_, REGION_NUM_, CONV_TYPE_, DOWN_RATIO_, DRCONV_POSITION_
        HIST_AS_GUIDE_ = hist_as_guide
        GUIDE_FEATURE_FROM_HIST_ = guide_feature_from_hist
        REGION_NUM_ = region_num
        CONV_TYPE_ = conv_type
        DOWN_RATIO_ = down_ratio
        DRCONV_POSITION_ = drconv_position

        if hist_conv_trainable:
            self.hist_conv1 = get_hist_conv(
                n_bins * in_channels, down_ratio, train=True)
            self.hist_conv2 = get_hist_conv(
                n_bins * in_channels, down_ratio, train=True)
            self.hist_conv3 = get_hist_conv(
                n_bins * in_channels, down_ratio, train=True)
        else:
            self.hist_conv = get_hist_conv(n_bins, down_ratio)

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(in_channels, C_NUMS[0])
        if hist_as_guide or guide_feature_from_hist or conv_type == 'dconv':
            extra_c_num = 0
        elif use_gray_hist:
            extra_c_num = n_bins
        else:
            extra_c_num = n_bins * in_channels

        if guide_feature_from_hist:
            kargs = {
                'guide_input_channel': n_bins if use_gray_hist else n_bins * in_channels
            }
        else:
            kargs = {}

        if encoder_use_hist:
            encoder_extra_c_num = extra_c_num
        else:
            encoder_extra_c_num = 0

        self.down1 = Down(C_NUMS[0] + encoder_extra_c_num,
                          C_NUMS[1], use_hist=encoder_use_hist)
        self.down2 = Down(C_NUMS[1] + encoder_extra_c_num,
                          C_NUMS[2], use_hist=encoder_use_hist)
        self.down3 = Down(C_NUMS[2] + encoder_extra_c_num,
                          C_NUMS[3], use_hist=encoder_use_hist)
        self.down4 = Down(C_NUMS[3] + encoder_extra_c_num,
                          C_NUMS[4] // factor, use_hist=encoder_use_hist)

        self.up1 = Up(C_NUMS[4] + extra_c_num,
                      C_NUMS[3] // factor, bilinear, **kargs)
        self.up2 = Up(C_NUMS[3] + extra_c_num,
                      C_NUMS[2] // factor, bilinear, **kargs)
        self.up3 = Up(C_NUMS[2] + extra_c_num,
                      C_NUMS[1] // factor, bilinear, **kargs)
        self.up4 = Up(C_NUMS[1] + extra_c_num, C_NUMS[0], bilinear, **kargs)
        self.outc = OutConv(C_NUMS[0], out_channels)

    def forward(self, x):
        # ipdb.set_trace()
        # get histograms
        # (`get_hist` return shape: n_bins, bs, c, h, w).

        if HIST_AS_GUIDE_ or self.use_gray_hist:
            histmap = get_hist(x, self.n_bins, grayscale=True)
        else:
            histmap = get_hist(x, self.n_bins)

        bs = x.shape[0]
        # out: [bs * c, n_bins, h, w]
        histmap = pack_tensor(histmap, self.n_bins).detach()
        if not self.hist_conv_trainable:
            hist_down2 = self.hist_conv(histmap)
            hist_down4 = self.hist_conv(hist_down2)
            hist_down8 = self.hist_conv(hist_down4)

            # [bs * c, b_bins, h, w] -> [bs, c*b_bins, h, w]
            for item in [histmap, hist_down2, hist_down4, hist_down8]:
                item.data = item.reshape(bs, -1, *item.shape[-2:])
        else:
            histmap = histmap.reshape(bs, -1, *histmap.shape[-2:])
            hist_down2 = self.hist_conv1(histmap)
            hist_down4 = self.hist_conv2(hist_down2)
            # [bs, n_bins * c, h/n, w/n]
            hist_down8 = self.hist_conv3(hist_down4)

        # forward
        encoder_hists = [None, ] * 4
        if self.encoder_use_hist:
            encoder_hists = [histmap, hist_down2, hist_down4, hist_down8]

        x1 = self.inc(x)
        x2 = self.down1(x1, encoder_hists[0])  # x2: 16
        x3 = self.down2(x2, encoder_hists[1])  # x3: 24
        x4 = self.down3(x3, encoder_hists[2])  # x4: 32
        x5 = self.down4(x4, encoder_hists[3])  # x5: 32

        # always apply hist in decoder:
        # ipdb.set_trace()
        x = self.up1(x5, x4, hist_down8)  # [x5, x4]: 32 + 32
        x = self.up2(x, x3, hist_down4)  # [x4, x3]:
        x = self.up3(x, x2, hist_down2)
        x = self.up4(x, x1, histmap)

        self.guide_features = [layer.guide_features for layer in [
            self.up1,
            self.up2,
            self.up3,
            self.up4,
        ]]

        logits = self.outc(x)
        return logits


class LowResHistUNet(HistUNet):
    def __init__(self, coeff_dim=12, opt=None):
        super(LowResHistUNet, self).__init__(
            in_channels=3,
            out_channels=coeff_dim * opt[LUMA_BINS],
            bilinear=True,
            **opt[HIST_UNET]
        )
        self.coeff_dim = coeff_dim
        print('[[ WARN ]] Using HistUNet in BilateralUpsampleNet as backbone')

    def forward(self, x):
        y = super(LowResHistUNet, self).forward(x)
        y = torch.stack(torch.split(y, self.coeff_dim, 1), 2)
        return y


class BilateralUpsampleNet(nn.Module):
    def __init__(self, opt):
        super(BilateralUpsampleNet, self).__init__()
        self.opt = opt
        global OPT
        OPT = opt
        self.guide = GuideNet(params=opt)
        self.slice = SliceNode(opt)
        self.build_coeffs_network(opt)

    def build_coeffs_network(self, opt):
        # Choose backbone:
        if opt[BACKBONE] == 'ori':
            Backbone = LowResNet
        elif opt[BACKBONE] == 'hist-unet':
            Backbone = LowResHistUNet
        else:
            raise NotImplementedError()

        # How to apply coeffs:
        # ───────────────────────────────────────────────────────────────────
        if opt[COEFFS_TYPE] == MATRIX:
            self.coeffs = Backbone(opt=opt)
            self.apply_coeffs = ApplyCoeffs()

        elif opt[COEFFS_TYPE] == GAMMA:
            print('[[ WARN ]] HDRPointwiseNN use COEFFS_TYPE: GAMMA.')

            # [ 008 ] change affine matrix -> other methods (alpha map, illu map)
            self.coeffs = Backbone(opt=opt, coeff_dim=24)
            self.apply_coeffs = ApplyCoeffsGamma()

        elif opt[COEFFS_TYPE] == 'retinex':
            print('[[ WARN ]] HDRPointwiseNN use COEFFS_TYPE: retinex.')
            self.coeffs = Backbone(opt=opt, coeff_dim=3)
            self.apply_coeffs = ApplyCoeffsRetinex()

        else:
            raise NotImplementedError(f'[ ERR ] coeff type {opt[COEFFS_TYPE]} unkown.')
        # ─────────────────────────────────────────────────────────────────────────────

    def forward(self, lowres, fullres):
        bilateral_grid = self.coeffs(lowres)
        try:
            self.guide_features = self.coeffs.guide_features
        except:
            ...
        guide = self.guide(fullres)
        self.guidemap = guide

        slice_coeffs = self.slice(bilateral_grid, guide)
        out = self.apply_coeffs(slice_coeffs, fullres)

        # use illu map:
        self.slice_coeffs = slice_coeffs
        # if self.opt[PREDICT_ILLUMINATION]:
        #
        #     power = self.opt[ILLU_MAP_POWER]
        #     if power:
        #         assert type(power + 0.1) == float
        #         out = out.pow(power)
        #
        #     out = out.clamp(fullres, torch.ones_like(out))
        #     # out = torch.where(out < fullres, fullres, out)
        #     self.illu_map = out
        #     out = fullres / (out + 1e-7)
        # else:
        self.illu_map = None

        if self.opt[PREDICT_ILLUMINATION]:
            return fullres / (out.clamp(fullres, torch.ones_like(out)) + 1e-7)
        else:
            return out


def get_gray(img):
    r = img[:, 0, ...]
    g = img[:, 1, ...]
    b = img[:, 2, ...]
    return (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(1)


def pack_tensor(x, n_bins):
    # pack tensor: transform gt_hist.shape [n_bins, bs, c, h, w] -> [bs*c, b_bins, h, w]
    # merge dim 1 (bs) and dim 2 (channel).
    return x.reshape(n_bins, -1, *x.shape[-2:]).permute(1, 0, 2, 3)


def get_hist(img, n_bins, grayscale=False):
    """
    Given a img (shape: bs, c, h, w),
    return the SOFT histogram map (shape: n_bins, bs, c, h, w)
                                or (shape: n_bins, bs, h, w) when grayscale=True.
    """
    if grayscale:
        img = get_gray(img)
    return torch.stack([
        torch.nn.functional.relu(
            1 - torch.abs(img - (2 * b - 1) / float(2 * n_bins)) * float(n_bins))
        for b in range(1, n_bins + 1)
    ])


def get_hist_conv(n_bins, kernel_size=2, train=False):
    """
    Return a conv kernel.
    The kernel is used to apply on the histogram map, shrinking the scale of the hist-map.
    """
    conv = torch.nn.Conv2d(n_bins, n_bins, kernel_size,
                           kernel_size, bias=False, groups=1)
    conv.weight.data.zero_()
    for i in range(conv.weight.shape[1]):
        alpha = kernel_size ** 2
        #         alpha = 1
        conv.weight.data[i, i, ...] = torch.ones(
            kernel_size, kernel_size) / alpha
    if not train:
        conv.requires_grad_(False)
    return conv


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DeepWBNet(nn.Module):
    def build_illu_net(self):

        return BilateralUpsampleNet(self.opt)

    def backbone_forward(self, net, x):

        low_x = self.down_sampler(x)
        res = net(low_x, x)
        try:
            self.res.update({'guide_features': net.guide_features})
        except:
            ...
            # print('[yellow]No guide feature found in BilateralUpsampleNet[/yellow]')
        return res

    def __init__(self, opt=None):
        super(DeepWBNet, self).__init__()
        self.opt = opt
        self.down_sampler = lambda x: F.interpolate(
            x, size=(256, 256), mode='bicubic', align_corners=False)
        self.illu_net = self.build_illu_net()

        # [ 021 ] use 2 illu nets (do not share weights).
        if not opt[SHARE_WEIGHTS]:
            self.illu_net2 = self.build_illu_net()

        # self.guide_net = GuideNN(out_channel=3)
        if opt[HOW_TO_FUSE] in ['cnn-weights', 'cnn-direct', 'cnn-softmax3']:
            # self.out_net = UNet(in_channels=9, wavelet=opt[USE_WAVELET])

            # [ 008-u1 ] use a simple network
            nf = 32
            self.out_net = nn.Sequential(
                nn.Conv2d(9, nf, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(nf, nf, 3, 1, 1),
                nn.ReLU(inplace=True),
                NONLocalBlock2D(nf, sub_sample='bilinear', bn_layer=False),
                nn.Conv2d(nf, nf, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(nf, 3, 1),
                NONLocalBlock2D(3, sub_sample='bilinear', bn_layer=False),
            )

        elif opt[HOW_TO_FUSE] in ['cnn-color']:
            # self.out_net = UNet(in_channels=3, wavelet=opt[USE_WAVELET])
            ...

        if not self.opt[BACKBONE_OUT_ILLU]:
            print('[[ WARN ]] Use output of backbone as brighten & darken directly.')
        self.res = {}

    def decomp(self, x1, illu_map):
        return x1 / (torch.where(illu_map < x1, x1, illu_map.float()) + 1e-7)

    def one_iter(self, x, attn_map, inverse_attn_map):
        # used only when USE_ATTN_MAP
        x1 = torch.cat((x, attn_map), 1)
        inverse_x1 = torch.cat((1 - x, inverse_attn_map), 1)

        illu_map = self.illu_net(x1, attn_map)
        inverse_illu_map = self.illu_net(inverse_x1)
        return illu_map, inverse_illu_map

    def forward(self, x):
        # ──────────────────────────────────────────────────────────
        # [ <008 ] use guideNN
        # x1 = self.guide_net(x).clamp(0, 1)

        # [ 008 ] use original input
        x1 = x
        inverse_x1 = 1 - x1

        if self.opt[USE_ATTN_MAP]:
            # [ 015 ] use attn map iteration to get illu map
            r, g, b = x[:, 0] + 1, x[:, 1] + 1, x[:, 2] + 1

            # init attn map as illumination channel of original input img:
            attn_map = (1. - (0.299 * r + 0.587 * g +
                              0.114 * b) / 2.).unsqueeze(1)
            inverse_attn_map = 1 - attn_map
            for _ in range(3):
                inverse_attn_map, attn_map = self.one_iter(
                    x, attn_map, inverse_attn_map)
            illu_map, inverse_illu_map = inverse_attn_map, attn_map

        elif self.opt[BACKBONE] == 'ynet':
            # [ 024 ] one encoder, 2 decoders.
            illu_map, inverse_illu_map = self.backbone_forward(
                self.illu_net, x1)

        else:
            illu_map = self.backbone_forward(self.illu_net, x1)
            if self.opt[SHARE_WEIGHTS]:
                inverse_illu_map = self.backbone_forward(
                    self.illu_net, inverse_x1)
            else:
                # [ 021 ] use 2 illu nets
                inverse_illu_map = self.backbone_forward(
                    self.illu_net2, inverse_x1)
        # ──────────────────────────────────────────────────────────

        if self.opt[BACKBONE_OUT_ILLU]:
            brighten_x1 = self.decomp(x1, illu_map)
            inverse_x2 = self.decomp(inverse_x1, inverse_illu_map)
        else:
            brighten_x1 = illu_map
            inverse_x2 = inverse_illu_map
        darken_x1 = 1 - inverse_x2
        # ──────────────────────────────────────────────────────────

        self.res.update({
            ILLU_MAP: illu_map,
            INVERSE_ILLU_MAP: inverse_illu_map,
            BRIGHTEN_INPUT: brighten_x1,
            DARKEN_INPUT: darken_x1,
        })

        # fusion:
        # ──────────────────────────────────────────────────────────
        if self.opt[HOW_TO_FUSE] == 'cnn-weights':
            # [ 009 ] only fuse 2 output image
            # fused_x = torch.cat([brighten_x1, darken_x1], dim=1)

            fused_x = torch.cat([x, brighten_x1, darken_x1], dim=1)

            # [ 007 ] get weight-map from UNet, then get output from weight-map
            weight_map = self.out_net(fused_x)  # <- 3 channels, [ N, 3, H, W ]
            w1 = weight_map[:, 0, ...].unsqueeze(1)
            w2 = weight_map[:, 1, ...].unsqueeze(1)
            w3 = weight_map[:, 2, ...].unsqueeze(1)
            out = x * w1 + brighten_x1 * w2 + darken_x1 * w3

            # [ 009 ] only fuse 2 output image
            # out = brighten_x1 * w1 + darken_x1 * w2
            # ────────────────────────────────────────────────────────────

        elif self.opt[HOW_TO_FUSE] == 'cnn-softmax3':
            fused_x = torch.cat([x, brighten_x1, darken_x1], dim=1)
            # <- 3 channels, [ N, 3, H, W ]
            weight_map = F.softmax(self.out_net(fused_x), dim=1)
            w1 = weight_map[:, 0, ...].unsqueeze(1)
            w2 = weight_map[:, 1, ...].unsqueeze(1)
            w3 = weight_map[:, 2, ...].unsqueeze(1)
            out = x * w1 + brighten_x1 * w2 + darken_x1 * w3

        # [ 006 ] get output directly from UNet
        elif self.opt[HOW_TO_FUSE] == 'cnn-direct':
            fused_x = torch.cat([x, brighten_x1, darken_x1], dim=1)
            out = self.out_net(fused_x)

        # [ 016 ] average 2 outputs.
        elif self.opt[HOW_TO_FUSE] == 'avg':
            out = 0.5 * brighten_x1 + 0.5 * darken_x1

        # [ 017 ] global color ajust
        elif self.opt[HOW_TO_FUSE] == 'cnn-color':
            out = 0.5 * brighten_x1 + 0.5 * darken_x1

        # elif self.opt[HOW_TO_FUSE] == 'cnn-residual':
        #     out = x +

        else:
            raise NotImplementedError(f'Unknown fusion method: {self.opt[HOW_TO_FUSE]}')

        return out


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample='pool', bn_layer=True):

        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            if sub_sample == 'pool':
                max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            elif sub_sample == 'bilinear':
                max_pool_layer = nn.UpsamplingBilinear2d([16, 16])
            else:
                raise NotImplementedError(f'[ ERR ] Unknown down sample method: {sub_sample}')
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample='pool', bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample='pool', bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer, )


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample='pool', bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer, )


class asign_index(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel, guide_feature):
        ctx.save_for_backward(kernel, guide_feature)
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True),
                                                              1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        return torch.sum(kernel * guide_mask, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        kernel, guide_feature = ctx.saved_tensors
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True),
                                                              1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        grad_kernel = grad_output.clone().unsqueeze(
            1) * guide_mask  # B x 3 x 256 x 25 x 25
        grad_guide = grad_output.clone().unsqueeze(1) * kernel  # B x 3 x 256 x 25 x 25
        grad_guide = grad_guide.sum(dim=2)  # B x 3 x 25 x 25
        softmax = F.softmax(guide_feature, 1)  # B x 3 x 25 x 25
        grad_guide = softmax * \
            (grad_guide - (softmax * grad_guide).sum(dim=1,
                                                     keepdim=True))  # B x 3 x 25 x 25
        return grad_kernel, grad_guide


def xcorr_slow(x, kernel, kwargs):
    """for loop to calculate cross correlation
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk, **kwargs)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel, kwargs):
    """group conv2d to calculate cross correlation
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, **kwargs, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


class Corr(Function):
    @staticmethod
    def symbolic(g, x, kernel, groups, kwargs):
        return g.op("Corr", x, kernel, groups_i=groups)

    @staticmethod
    def forward(self, x, kernel, groups, kwargs):
        """group conv2d to calculate cross correlation
        """
        batch = x.size(0)
        channel = x.size(1)
        x = x.view(1, -1, x.size(2), x.size(3))
        kernel = kernel.view(-1, channel // groups,
                             kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, **kwargs, groups=groups * batch)
        out = out.view(batch, -1, out.size(2), out.size(3))
        return out


class Correlation(nn.Module):
    use_slow = True

    def __init__(self, use_slow=None):
        super(Correlation, self).__init__()
        if use_slow is not None:
            self.use_slow = use_slow
        else:
            self.use_slow = Correlation.use_slow

    def extra_repr(self):
        if self.use_slow:
            return "xcorr_slow"
        return "xcorr_fast"

    def forward(self, x, kernel, **kwargs):
        if self.training:
            if self.use_slow:
                return xcorr_slow(x, kernel, kwargs)
            else:
                return xcorr_fast(x, kernel, kwargs)
        else:
            return Corr.apply(x, kernel, 1, kwargs)


class DRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=8, guide_input_channel=False, **kwargs):
        super(DRConv2d, self).__init__()
        self.region_num = region_num
        self.guide_input_channel = guide_input_channel

        if in_channels == 64:
            inp_size = 32
        elif in_channels == 32:
            inp_size = 64
        elif in_channels == 16:
            inp_size = 128
        elif in_channels == 8:
            inp_size = 256
        else:
            raise "Svertki slomalis"
        conv_kernel_stride = inp_size // kernel_size
        conv_kernel_kernel_size = inp_size - (kernel_size-1)*conv_kernel_stride

        self.conv_kernel = nn.Sequential(
            nn.AvgPool2d(conv_kernel_kernel_size, stride=conv_kernel_stride),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1,
                      groups=region_num)
        )
        if guide_input_channel:
            # get guide feature from a user input tensor.
            self.conv_guide = nn.Conv2d(
                guide_input_channel, region_num, kernel_size=kernel_size, **kwargs)
        else:
            self.conv_guide = nn.Conv2d(
                in_channels, region_num, kernel_size=kernel_size, **kwargs)

        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.asign_index = asign_index.apply

    def forward(self, input, guide_input=None):
        kernel = self.conv_kernel(input)
        # kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3))  # B x (r*in*out) x W X H
        output = self.corr(input, kernel, **self.kwargs)  # B x (r*out) x W x H
        output = output.view(output.size(0), self.region_num, -1,
                             output.size(2), output.size(3))  # B x r x out x W x H
        if self.guide_input_channel:
            guide_feature = self.conv_guide(guide_input)
        else:
            guide_feature = self.conv_guide(input)
        self.guide_feature = guide_feature
        # self.guide_feature = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True), 1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        output = self.asign_index(output, guide_feature)
        return output


class HistDRConv2d(DRConv2d):
    def forward(self, input, histmap):
        """
        use histmap as guide feature directly.
        histmap.shape: [bs, n_bins, h, w]
        """
        histmap.requires_grad_(False)

        kernel = self.conv_kernel(input)
        output = self.corr(input, kernel, **self.kwargs)  # B x (r*out) x W x H
        output = output.view(output.size(0), self.region_num, -1,
                             output.size(2), output.size(3))  # B x r x out x W x H
        output = self.asign_index(output, histmap)
        return output

LOGGER_BUFFER_LOCK = False
SPLIT = '————————————————————————————————————————————————————'

GLOBAL_SEED = 233
TEST_RESULT_DIRNAME = 'test_result'
TRAIN_LOG_DIRNAME = 'log'
CONFIG_DIR = 'config'
CONFIG_FILEPATH = 'config/config.yaml'
OPT_FILENAME = 'CONFIG.yaml'
LOG_FILENAME = 'run.log'
LOG_TIME_FORMAT = '%Y-%m-%d_%H:%M:%S'
INPUT = 'input'
OUTPUT = 'output'
GT = 'GT'
STRING_FALSE = 'False'
SKIP_FLAG = 'q'
DEFAULTS = 'defaults'
HYDRA = 'hydra'

INPUT_FPATH = 'input_fpath'
GT_FPATH = 'gt_fpath'

DEBUG = 'debug'
BACKEND = 'backend'
CHECKPOINT_PATH = 'checkpoint_path'
LOG_DIRPATH = 'log_dirpath'
IMG_DIRPATH = 'img_dirpath'
DATALOADER_N = 'dataloader_num_worker'
VAL_DEBUG_STEP_NUMS = 'val_debug_step_nums'
VALID_EVERY = 'valid_every'
LOG_EVERY = 'log_every'
AUGMENTATION = 'aug'
RUNTIME_PRECISION = 'runtime_precision'
NUM_EPOCH = 'num_epoch'
NAME = 'name'
LOSS = 'loss'
TRAIN_DATA = 'train_ds'
VALID_DATA = 'valid_ds'
TEST_DATA = 'test_ds'
GPU = 'gpu'
RUNTIME = 'runtime'
CLASS = 'class'
MODELNAME = 'modelname'
BATCHSIZE = 'batchsize'
VALID_BATCHSIZE = 'valid_batchsize'
LR = 'lr'
CHECKPOINT_MONITOR = 'checkpoint_monitor'
MONITOR_MODE = 'monitor_mode'
COMMENT = 'comment'
EARLY_STOP = 'early_stop'
AMP_BACKEND = 'amp_backend'
AMP_LEVEL = 'amp_level'
VALID_RATIO = 'valid_ratio'

LTV_LOSS = 'ltv'
COS_LOSS = 'cos'
SSIM_LOSS = 'ssim_loss'
L1_LOSS = 'l1_loss'
COLOR_LOSS = 'l_color'
SPATIAL_LOSS = 'l_spa'
EXPOSURE_LOSS = 'l_exp'
WEIGHTED_LOSS = 'weighted_loss'
PSNR_LOSS = 'psnr_loss'
HIST_LOSS = 'hist_loss'
INTER_HIST_LOSS = 'inter_hist_loss'
VGG_LOSS = 'vgg_loss'

PSNR = 'psnr'
SSIM = 'ssim'

VERTICAL_FLIP = 'v-flip'
HORIZON_FLIP = 'h-flip'
DOWNSAMPLE = 'downsample'
RESIZE_DIVISIBLE_N = 'resize_divisible_n'
CROP = 'crop'
LIGHTNESS_ADJUST = 'lightness_adjust'
CONTRAST_ADJUST = 'contrast_adjust'

BUNET = 'bilateral_upsample_net'
UNET = 'unet'
HIST_UNET = 'hist_unet'
PREDICT_ILLUMINATION = 'predict_illumination'
FILTERS = 'filters'

MODE = 'mode'
COLOR_SPACE = 'color_space'
BETA1 = 'beta1'
BETA2 = 'beta2'
LAMBDA_SMOOTH = 'lambda_smooth'
LAMBDA_MONOTONICITY = 'lambda_monotonicity'
MSE = 'mse'
L2_LOSS = 'l2_loss'
TV_CONS = 'tv_cons'
MN_CONS = 'mv_cons'
WEIGHTS_NORM = 'wnorm'
TEST_PTH = 'test_pth'

LUMA_BINS = 'luma_bins'
CHANNEL_MULTIPLIER = 'channel_multiplier'
SPATIAL_BIN = 'spatial_bin'
BATCH_NORM = 'batch_norm'
NET_INPUT_SIZE = 'net_input_size'
LOW_RESOLUTION = 'low_resolution'
ONNX_EXPORTING_MODE = 'onnx_exporting_mode'
SELF_SUPERVISED = 'self_supervised'
COEFFS_TYPE = 'coeffs_type'
ILLU_MAP_POWER = 'illu_map_power'
GAMMA = 'gamma'
MATRIX = 'matrix'
GUIDEMAP = 'guidemap'
USE_HSV = 'use_hsv'

USE_WAVELET = 'use_wavelet'
NON_LOCAL = 'use_non_local'
USE_ATTN_MAP = 'use_attn_map'
ILLUMAP_CHANNEL = 'illumap_channel'
HOW_TO_FUSE = 'how_to_fuse'
SHARE_WEIGHTS = 'share_weights'
BACKBONE = 'backbone'
ARCH = 'arch'
N_BINS = 'n_bins'
BACKBONE_OUT_ILLU = 'backbone_out_illu'
CONV_TYPE = 'conv_type'
HIST_AS_GUIDE_ = 'hist_as_guide'
ENCODER_USE_HIST = 'encoder_use_hist'
GUIDE_FEATURE_FROM_HIST = 'guide_feature_from_hist'
NC = 'channel_nums'

ILLU_MAP = 'illu_map'
INVERSE_ILLU_MAP = 'inverse_illu_map'
BRIGHTEN_INPUT = 'brighten_input'
DARKEN_INPUT = 'darken_input'

TRAIN = 'train'
TEST = 'test'
VALID = 'valid'
ONNX = 'onnx'
CONDOR = 'condor'
IMAGES = 'images'

conf = {
    "hist_unet": {
        'n_bins': 8,
        'hist_as_guide': False,
        'channel_nums': [8, 16, 32, 64, 128],
        'encoder_use_hist': False,
        'guide_feature_from_hist': True,
        'region_num': 2,
        'use_gray_hist': False,
        'conv_type': 'drconv',
        'down_ratio': 2,
        'hist_conv_trainable': False,
        'drconv_position': [0, 1]
    },
    'predict_illumination': False,
    'luma_bins': 8,
    'channel_multiplier': 1,
    'spatial_bin': 16,
    'batch_norm': True,
    'low_resolution': 256,
    'coeffs_type': 'matrix',
    'conv_type': 'conv',
    'illu_map_power': False,
    'use_wavelet': False,
    'use_attn_map': False,
    'use_non_local': False,
    'how_to_fuse': 'cnn-weights',
    'backbone': 'hist-unet',
    'backbone_out_illu': True,
    'illumap_channel': 3,
    'share_weights': True,
    'n_bins': 8,
    'hist_as_guide': False
}


'''
def main():

    weights = torch.load("trained_on_MSEC.ckpt")["state_dict"]

    for w in list(weights.keys()):
        weights[w[4:]] = weights.pop(w)

    weights.pop("loss.hist_conv.weight")
    weights.pop("r_histloss.hist_conv.weight")

    model = DeepWBNet(conf)
    model.load_state_dict(weights)
    model.to("cuda")
    model.eval()

    img = cv2.imread("tests/IMG_2416.png")
    inp_size = img.shape[:2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))

    img = to_tensor(img).unsqueeze(0).to("cuda")

    with torch.no_grad():
        res = (model(img)*255)[0].permute((1, 2, 0))
    res = res.cpu().detach().numpy()
    res = res.clip(0, 255).astype(np.uint8)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    res = cv2.resize(res, inp_size[::-1])

    cv2.imwrite("res/IMG_2416.png", res)
'''


class TritonPythonModel:
    def initialize(self, args):
        weights = torch.load("/assets/trained_on_MSEC.ckpt")["state_dict"]

        for w in list(weights.keys()):
            weights[w[4:]] = weights.pop(w)

        weights.pop("loss.hist_conv.weight")
        weights.pop("r_histloss.hist_conv.weight")

        self.model = DeepWBNet(conf)
        self.model.load_state_dict(weights)

    def predict(self, img):
        img = to_tensor(img).unsqueeze(0)
        with torch.no_grad():
            res = (self.model(img)*255).permute((0, 2, 3, 1)).cpu().detach().numpy()
        return res

    def execute(self, requests):
        responses = []
        for request in requests:
            imgs = pb_utils.get_input_tensor_by_name(request, "input").as_numpy()

            tokens = self.predict(imgs)

            output_tensor_tokens = pb_utils.Tensor("output", tokens)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor_tokens]
            )
            responses.append(inference_response)
        return responses