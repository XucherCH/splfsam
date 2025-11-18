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

class Unet(nn.Module):
    def __init__(self,in_ch,out_ch,mid_ch):
        super(Unet,self).__init__()

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
        # Residual connection branch
        x_shortcut = self.shortcut_conv(x)
        
        # Down-sampling path
        h1 = self.conv1(x)               # First layer
        hx = self.downsample1(h1)         # Down-sampling
        
        h2 = self.conv2(hx)               # Second layer
        hx = self.downsample2(h2)         # Down-sampling
        
        h3 = self.conv3(hx)               # Third layer
        hx = self.downsample3(h3)         # Down-sampling
        
        h4 = self.conv4(hx)               # Fourth layer
        hx = self.downsample4(h4)         # Down-sampling
        
        h5 = self.conv5(hx)               # Fifth layer
        h6 = self.conv6(h5)               # Bottleneck layer
        
        # Up-sampling path (with skip connections)
        hx5d = self.conv5d(torch.cat((h6, h5), 1))  # Merge bottleneck layer and fifth layer
        hx5d_up = _upsample_like(hx5d, h4)          # Up-sample to fourth layer size
        
        hx4d = self.conv4d(torch.cat((hx5d_up, h4), 1)) 
        hx4d_up = _upsample_like(hx4d, h3)
        
        hx3d = self.conv3d(torch.cat((hx4d_up, h3), 1))
        hx3d_up = _upsample_like(hx3d, h2)
        
        hx2d = self.conv2d(torch.cat((hx3d_up, h2), 1))
        hx2d_up = _upsample_like(hx2d, h1)
        
        hx1d = self.conv1d(torch.cat((hx2d_up, h1), 1))
        
        # Residual connection
        return hx1d + x_shortcut
    
if __name__ == "__main__":

    model = Unet(32,32,16).cuda()

    x = torch.randn(4,32,128,128).cuda()
    y = model(x)
    print(y.shape)