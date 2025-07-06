"""
Created on Thu Apr 16 20:31:28 2024
@author: laxio (Chuang Liu)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from thop import profile
from einops import rearrange

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    # print('---------------- LL ------------------')
    # print(x_LL[:,0,:,:])
    # print()
    # print('---------------- HL ------------------')
    # print(x_HL[:,0,:,:])
    # print()
    # print('---------------- LH ------------------')
    # print(x_LH[:,0,:,:])
    # print()
    # print('---------------- HH ------------------')
    # print(x_HH[:,0,:,:])
    return x_LL, x_HL, x_LH, x_HH


# 使用哈尔 haar 小波变换来实现二维逆向离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    # print('-------------- enter iwt ---------------')
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    # print('-------------- back ---------------')
    return h

# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)

# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.conv1 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        rs1 = self.relu(self.conv0(x))
        rs1 = self.conv1(rs1)
        rs = torch.add(x, rs1)
        return rs

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):  # i: 0,1; n: 64, 64
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)  # (1-(-1))/(2*64)=1/64
        seq = v0 + r + (2 * r) * torch.arange(n).float()  # -1//1 + 1/64 + 2/64 * [0, 1, 2, ...,63]   (1,64)
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)  # (2, 64, 64)->(64, 64, 2)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])  # (64*64, 2)
    return ret


class Network(nn.Module):

    def __init__(self, args, scale_ratio=4, n_select_bands=1, n_bands=8, image_size=64, feat_dim=64,
                 guide_dim=64, H=64, W=64, mlp_dim=[256, 128], NIR_dim=33):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim
        self.NIR_dim = NIR_dim
        self.image_size = image_size
        self.scale_ratio = scale_ratio
        self.n_select_bands = n_select_bands
        self.n_bands = n_bands
        # self.encoder32 = nn.AdaptiveMaxPool2d((H // 4, W // 4))  # down*4
        self.encoder64 = nn.AdaptiveMaxPool2d((H // 2, W // 2))  # down*2

        self.blk_1_32 = nn.Sequential(
            nn.Conv2d(n_select_bands, feat_dim, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.blk_8_32 = nn.Sequential(
            nn.Conv2d(n_bands, feat_dim, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.DE_pan = ResBlock(in_channels=feat_dim, hidden_channels=feat_dim, out_channels=feat_dim)
        self.DE_lms = ResBlock(in_channels=feat_dim, hidden_channels=feat_dim, out_channels=feat_dim)
        self.dwt_pan = DWT()
        self.dwt_lms = DWT()
        self.DE_HL = ResBlock(in_channels=feat_dim, hidden_channels=feat_dim, out_channels=feat_dim)
        self.DE_LH = ResBlock(in_channels=feat_dim, hidden_channels=feat_dim, out_channels=feat_dim)
        self.DE_HH = ResBlock(in_channels=feat_dim, hidden_channels=feat_dim, out_channels=feat_dim)

        # self.DE_HL_LH = ResBlock(60, 30, 30)
        self.DE_HH_HH = ResBlock(in_channels=feat_dim, hidden_channels=feat_dim, out_channels=feat_dim)
        self.iwt = IWT()
        self.DE_PAN = ResBlock(in_channels=feat_dim, hidden_channels=feat_dim, out_channels=feat_dim)

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(feat_dim//4, feat_dim, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.spectral_encoder = nn.Sequential(
            nn.Conv2d(self.n_bands, feat_dim, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

        imnet_in_dim_64 = self.feat_dim + self.guide_dim + 2  # 512+2
        imnet_in_dim_128 = NIR_dim - 1 + self.guide_dim + 2  # 128-1+256+2

        self.imnet_64 = MLP(imnet_in_dim_64, out_dim=NIR_dim, hidden_list=self.mlp_dim)
        self.imnet_128 = MLP(imnet_in_dim_128, out_dim=n_bands+1, hidden_list=self.mlp_dim)


    def query_64(self, feat, coord, hr_guide):

        b, c, h, w = feat.shape
        _, _, H, W = hr_guide.shape
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h,
                                                                                                            w).cuda()
        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()  # [B, N, 2]

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)

                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)

                rel_coord = coord - q_coord  # [B, N, 2]
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)

                pred = self.imnet_64(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 4]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 4, 4]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(
            -1)  # （B, N, 3, 4）->(B, N, 3，1)->(B, N, 3)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)  # (B, 3, N)->(B, 3, 32, 32)

        return ret

    def query_128(self, feat, coord, hr_guide):  # ((B， NIR_dim, 32, 32), (64*64, 2), (B, 64, 64, 64))->(B, 8, 64, 64)

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h,
                                                                                                            w).cuda()

        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)

                pred = self.imnet_128(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def forward(self, pan, lms, ms):  # (B, 8, 16, 16)  (B, 1, 64, 64)
        p_LL, p_HL, p_LH, p_HH = self.dwt_pan(self.DE_pan(self.blk_1_32(pan)))
        m_LL, m_HL, m_LH, m_HH = self.dwt_lms(self.DE_lms(self.blk_8_32(lms)))
        f_HL = self.DE_HL(torch.add(p_HL, m_HL))
        f_LH = self.DE_LH(torch.add(p_LH, m_LH))
        f_HH = self.DE_HH(torch.add(p_HH, m_HH))

        f_hp = self.DE_HH_HH(torch.add(f_HL, f_LH))
        f_hp = self.DE_PAN(torch.add(f_hp, f_HH))  # 256
        D = self.iwt(torch.add(f_hp, p_LL))  # 64

        _, _, H, W = pan.shape  # (64, 64)

        coord_64 = make_coord([H // 2, W // 2]).cuda()  # (32*32, 2)
        coord_128 = make_coord([H, W]).cuda()  # (64*64, 2)

        hr_spa = self.spatial_encoder(D.cuda())  #256
        guide64 = self.encoder64(hr_spa)  # (B, 256, 32, 32)

        lr_spe = self.spectral_encoder(ms)

        sr_2 = self.query_64(lr_spe, coord_64, guide64)

        sr = self.query_128(sr_2, coord_128, hr_spa)

        output = lms + sr

        return output

from torchsummary import summary
def summaries(model, grad=False):
    if grad:
        summary(model, input_size=[(1, 64, 64), (4, 64, 64), (4, 16, 16)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)