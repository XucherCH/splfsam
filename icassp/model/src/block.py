import torch
from torch import nn
# from model.src.common import LayerNorm2d
from torch.nn import functional as F
from einops import rearrange, repeat

class MAFA(nn.Module):
    def __init__(self, dim,patch_size=8, bias=False):
        super(MAFA, self).__init__()

        self.patch_size = patch_size

        self.dim = dim


        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_in = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):

        x = self.project_in(x)

        b, c, h, w = x.shape
        h_n = (8 - h % 8) % 8
        w_n = (8 - w % 8) % 8
        
        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        
        x=x[:,:,:h,:w]
        
        return x
    




class MLFusion(nn.Module):
    def __init__(self, norm, act):
        super().__init__()
        self.fusi_conv = nn.Sequential(
            nn.Conv2d(1024, 256, 1,bias = False),
            norm(256),
            act(),
        )

        self.attn_conv = nn.ModuleList()
        for i in range(4):
            self.attn_conv.append(nn.Sequential(
                nn.Conv2d(256, 256, 1,bias = False),
                norm(256),
                act(),
            ))

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_list):
        fusi_feature = torch.cat(feature_list, dim = 1).contiguous()
        fusi_feature = self.fusi_conv(fusi_feature)

        for i in range(4):
            x = feature_list[i]
            attn = self.attn_conv[i](x)
            attn = self.pool(attn)
            attn = self.sigmoid(attn)

            x = attn * x + x
            feature_list[i] = x
        
        return feature_list[0] , feature_list[1] , feature_list[2] , feature_list[3]
    
    
