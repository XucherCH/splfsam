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

        # 修改1: 将输入通道数改为 dim * height
        self.mfb = MFB(dim * height)  # 注意这里传入的是拼接后的通道数

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        
        # 拼接特征图
        in_feats = torch.cat(in_feats, dim=1)  # 形状变为 [B, C*height, H, W]
        
        # 修改2: 移除多余的 num 参数
        in_feats = self.mfb(in_feats)  # 仅传递一个参数
        
        # 重塑为多特征图形式
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

    def forward(self, x):  # 只接受一个参数
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
    x3 = torch.randn(4, 16, 128, 128).cuda()
    x4 = torch.randn(4, 16, 128, 128).cuda()
    x5 = torch.randn(4, 16, 128, 128).cuda()
    my_list = [x1, x2,x3,x4,x5]
    
    model = UMFEB(16,5).cuda()
    y = model(my_list)
    print(y.shape)  # 应该输出: torch.Size([4, 16, 128, 128])