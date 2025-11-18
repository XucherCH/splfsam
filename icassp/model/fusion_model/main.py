import torch
import torch.nn as nn

class MFM(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(MFM, self).__init__()

        self.height = height
        d = max(int(dim/reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim*height, 1, bias=False)
        )

        # Modification 1: Change the input channel number to dim * height
        self.mfb = MFB(dim * height)  # Note that the concatenated channel number is passed in here

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        
        # Concatenate feature maps
        in_feats = torch.cat(in_feats, dim=1)  # Shape becomes [B, C*height, H, W]
        
        # Modification 2: Remove the redundant num parameter
        in_feats = self.mfb(in_feats)  # Only pass one parameter
        
        # Reshape to multi-feature map form
        in_feats = in_feats.view(B, self.height, C, H, W)
        
        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out

class MFB(nn.Module):  # Multiscale Fusion Block
    def __init__(self, in_c):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0, groups=in_c),
            nn.ReLU()
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1, groups=in_c),
            nn.ReLU()
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=5, stride=1, padding=2, groups=in_c),
            nn.ReLU()
        )

        self.conv_7 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=7, stride=1, padding=3, groups=in_c),
            nn.ReLU()
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(in_c * 4, in_c, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):  # Only accepts one parameter
        x_shortcut = x
        x = self.conv_in(x)
        x1 = self.conv_1(x)
        x3 = self.conv_3(x)
        x5 = self.conv_5(x)
        x7 = self.conv_7(x)

        xout = torch.cat([x1, x3, x5, x7], dim=1)
        xout = self.conv_out(xout)

        return x_shortcut + xout
    


class UMFEB(nn.Module):

    def __init__(self,dim,height=2):
        super().__init__()

        self.conv_last = MFB(dim)

        self.conv_fusion = MFM(dim,height)

        self.conv_q = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )

        self.conv_k = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )

        self.conv_v = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )

        self.conv_atten = nn.Sequential(
            nn.Conv2d(dim,1,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )

    def forward(self,x:list):

        new_img = x[-1]
        new1 = self.conv_last(new_img)

        fusion = self.conv_fusion(x)

        q = self.conv_q(new1)
        k = self.conv_k(fusion)

        v = self.conv_v(new1)

        result = q@k@v
      

        atten = self.conv_atten(fusion)

        return atten*result + new_img



if __name__ == "__main__":
    x1 = torch.randn(4, 16, 128, 128).cuda()
    x2 = torch.randn(4, 16, 128, 128).cuda()
    my_list = [x1, x2]
    
    model = UMFEB(16).cuda()
    y = model(my_list)
    print(y.shape)  # Should output: torch.Size([4, 16, 128, 128])