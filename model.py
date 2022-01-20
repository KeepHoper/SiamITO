import torch
import torch.nn as nn
import torch.nn.functional as fct
from torch import randn, cat, matmul


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=int(in_ch/2), kernel_size=1),
            nn.Conv2d(in_channels=int(in_ch/2), out_channels=out_ch, kernel_size=3)
            )

    def forward(self, x):
        res = self.conv(x)
        return res


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
        )

        self.res1 = ResidualBlock(32, 64)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
        )

        self.res2 = ResidualBlock(128, 128)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1)
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        res1 = self.res1(conv1)
        conv2 = self.conv2(conv1)
        conv2 = cat((conv2, res1), dim=1)
        conv3 = self.conv3(conv2)
        res2 = self.res2(conv3)
        conv4 = self.conv4(conv3)
        conv4 = cat((conv4, res2), dim=1)
        output = self.conv5(conv4)

        # output = fct.interpolate(conv5, conv2.shape[3], mode='bilinear', align_corners=False)

        return output


class SiameseITO(nn.Module):
    def __init__(self, embedding_net, upscale=False, upscale_size=None):
        super(SiameseITO, self).__init__()
        self.embedding_net = embedding_net
        self.match_BatchNorm = nn.BatchNorm2d(1)
        self.upscale = upscale
        self.upscale_size = upscale_size
        self.self_attn = SelfAtt(in_channels=128)
        self.cross_attn = CrossAtt(in_channels=128)

        self.adjust_attn = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x1, x2):
        embedding_template = self.embedding_net(x1)
        embedding_search = self.embedding_net(x2)

        self_attn_template = self.self_attn(embedding_template)
        self_attn_search = self.self_attn(embedding_search)
        cross_attn_template = self.cross_attn(embedding_template, embedding_search)
        cross_attn_search = self.cross_attn(embedding_search, embedding_template)

        attn_template = cat((self_attn_template, cross_attn_template), dim=1)
        attn_search = cat((self_attn_search, cross_attn_search), dim=1)

        attn_template = self.adjust_attn(attn_template)
        attn_search = self.adjust_attn(attn_search)

        match_map = self.match_corr(attn_template, attn_search)

        return match_map

    def match_corr(self, embed_tem, embed_srh):
        b, c, h, w = embed_srh.shape

        match_map = fct.conv2d(embed_srh.view(1, b * c, h, w), embed_tem, stride=1, groups=b)

        match_map = match_map.permute(1, 0, 2, 3)
        match_map = self.match_BatchNorm(match_map)

        if self.upscale:
            match_map = fct.interpolate(match_map, self.upscale_size, mode='bilinear', align_corners=False)

        return match_map

    def conv(self, x):
        x = self.embedding_net(x)
        return x


class BCEWeightLoss(nn.Module):
    def __init__(self):
        super(BCEWeightLoss, self).__init__()

    def forward(self, input, target, weight=None):
        return fct.binary_cross_entropy_with_logits(input, target, weight, reduction='sum')


class SelfAtt(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(SelfAtt, self).__init__()
        self.dimension = 2
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

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
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = matmul(theta_x, phi_x)
        f_div_C = fct.softmax(f, dim=-1)

        y = matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class CrossAtt(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(CrossAtt, self).__init__()

        self.in_channels = in_channels
        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d

        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

    def forward(self, main_feature, cross_feature):
        batch_size = main_feature.size(0)
        main_size = main_feature.size(2)
        cross_size = cross_feature.size(2)

        x = self.g(cross_feature).view(batch_size, self.inter_channels, -1)
        x = x.permute(0, 2, 1)

        y = self.theta(cross_feature).view(batch_size, self.inter_channels, -1)

        z = fct.interpolate(main_feature, cross_size, mode='bilinear', align_corners=False)
        z = self.phi(z).view(batch_size, self.inter_channels, -1)
        z = z.permute(0, 2, 1)

        f = torch.matmul(x, y)
        f_div_C = fct.softmax(f, dim=-1)

        output = torch.matmul(f_div_C, z)
        output = output.permute(0, 2, 1).contiguous()
        output = output.view(batch_size, self.inter_channels, *cross_feature.size()[2:])

        output = self.W(output)
        output = fct.interpolate(output, main_size, mode='bilinear', align_corners=False)
        output += main_feature
        return output

