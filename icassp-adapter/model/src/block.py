import torch
from torch import nn
# from model.src.common import LayerNorm2d
from torch.nn import functional as F
from einops import rearrange, repeat

class FFT(nn.Module):
    def __init__(self, dim,patch_size=8, bias=False):
        super(FFT, self).__init__()

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
    
    
class FFTFusion(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(FFTFusion, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(

                nn.Conv2d(in_dim, reduction_dim, kernel_size=bin,padding=bin//2),
                nn.GELU(),
                nn.Conv2d(reduction_dim, reduction_dim, kernel_size=1, bias=False),
                nn.GELU(),

                FFT(reduction_dim,8)
            ))
        self.features = nn.ModuleList(self.features)
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding = 1, bias=False, groups = in_dim),
            nn.GELU(),
        )
        

    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class MAFA(nn.Module):
    def __init__(self, in_dim, hidden_dim, patch_num):
        super().__init__()
        self.down_project = nn.Linear(in_dim,hidden_dim)
        self.act = nn.GELU()
        self.mppm = FFTFusion(hidden_dim, hidden_dim //4,  [1,3,5,7])
        self.patch_num = patch_num
        self.up_project = nn.Linear(hidden_dim, in_dim)
        self.down_conv = nn.Sequential(nn.Conv2d(hidden_dim*2, hidden_dim, 1),
                                       nn.GELU())

    def forward(self, x):
        down_x = self.down_project(x)
        down_x = self.act(down_x)

        down_x = down_x.permute(0, 3, 1, 2).contiguous()
        down_x = self.mppm(down_x).contiguous()
        down_x = self.down_conv(down_x)
        down_x = down_x.permute(0, 2, 3, 1).contiguous()

        up_x = self.up_project(down_x)
        return x + up_x

if __name__ == "__main__":

    x = torch.randn(4,32,32,256*3).cuda()
    model = MAFA(256*3,256,4).cuda()

    y = model(x)
    print(y.shape)