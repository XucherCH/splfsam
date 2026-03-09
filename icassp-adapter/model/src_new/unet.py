import torch
import torch.nn as nn
import torch.nn.functional as F


class CBR(nn.Module): 
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(CBR,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout
    
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

    

def _upsample_like(src,tar):
    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')
    return src

class UNET(nn.Module): # U-shaped Multiscale Mask Feature Embedding Block
    def __init__(self,in_ch,out_ch,mid_ch):
        super(UNET,self).__init__()

        self.shortcut_conv = CBR(in_ch,out_ch)

        self.conv1 = CBR(in_ch,mid_ch)                                                    
        self.downsample1 = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True) #64

        self.conv2 = CBR(mid_ch,mid_ch)                                                    
        self.downsample2 = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True) #32

        self.conv3 = CBR(mid_ch,mid_ch)                                                    
        self.downsample3 = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True) #16

        self.conv4 = CBR(mid_ch,mid_ch)                                                    
        self.downsample4 = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True) #8

        self.conv5 = CBR(mid_ch,mid_ch)   
        self.conv6 = CBR(mid_ch,mid_ch)   

        self.conv5d = CBR(mid_ch*2,mid_ch)
        self.conv4d = CBR(mid_ch*2,mid_ch)
        self.conv3d = CBR(mid_ch*2,mid_ch)
        self.conv2d = CBR(mid_ch*2,mid_ch)
        self.conv1d = CBR(mid_ch*2,out_ch)

    def forward(self,x):
        # 残差连接分支
        x_shortcut = self.shortcut_conv(x)
        
        # 下采样路径
        h1 = self.conv1(x)               # 第一层
        hx = self.downsample1(h1)         # 下采样
        
        h2 = self.conv2(hx)               # 第二层
        hx = self.downsample2(h2)         # 下采样
        
        h3 = self.conv3(hx)               # 第三层
        hx = self.downsample3(h3)         # 下采样
        
        h4 = self.conv4(hx)               # 第四层
        hx = self.downsample4(h4)         # 下采样
        
        h5 = self.conv5(hx)               # 第五层
        h6 = self.conv6(h5)               # 瓶颈层
        
        # 上采样路径（带跳跃连接）
        hx5d = self.conv5d(torch.cat((h6, h5), 1))  # 合并瓶颈层和第五层
        hx5d_up = _upsample_like(hx5d, h4)          # 上采样到第四层尺寸
        
        hx4d = self.conv4d(torch.cat((hx5d_up, h4), 1)) 
        hx4d_up = _upsample_like(hx4d, h3)
        
        hx3d = self.conv3d(torch.cat((hx4d_up, h3), 1))
        hx3d_up = _upsample_like(hx3d, h2)
        
        hx2d = self.conv2d(torch.cat((hx3d_up, h2), 1))
        hx2d_up = _upsample_like(hx2d, h1)
        
        hx1d = self.conv1d(torch.cat((hx2d_up, h1), 1))
        
        # 残差连接
        return hx1d + x_shortcut
    
if __name__ == "__main__":

    model = UNET(32,32,16).cuda()

    x = torch.randn(4,32,128,128).cuda()
    y = model(x)
    print(y.shape)