import torch.nn as nn
import torch
from torch.autograd import Variable
from padding_same_conv import Conv2d
def toTensor(image):
    return torch.from_numpy(image.transpose((0,3,1,2)))

def tensor_to_np(var):
    return var.data.cpu().numpy()
class ConvLayer(nn.Sequential):
    def __init__(self,in_channels,out_channels):
        super(ConvLayer,self).__init__()
        self.add_module('conv2d_',Conv2d(in_channels,out_channels,kernel_size=5,stride=2))
        self.add_module('leakyrelu',nn.LeakyReLU(0.1,inplace=True))
        #self.add_module('pixelshuffle',PixelShuffle())

##create conv,leakyrelu,pixel shuffle(b,c,h,w——>b,c//4,h//2,2,w//2,2——>b,c//4,h*2,w*2)
class UpScale(nn.Sequential):
    def __init__(self,in_channels,out_channels):
        super(UpScale,self).__init__()
        self.add_module('conv2d_',Conv2d(in_channels,out_channels*4,kernel_size=3))
        self.add_module('leakyrelu',nn.LeakyReLU(0.1,inplace=True))
        self.add_module('pixelshuffle',PixelShuffle())
##flatten for fully connection layer
class Flatten(nn.Module):
    def forward(self,input):
        return input.view(input.size(0),-1)

##reshape for decoder
class Reshape(nn.Module):
    def forward(self,input):
        return input.view(-1,1024,4,4)

class PixelShuffle(nn.Module):
    def forward(self, input):
        batch_size, c, h, w = input.size()
        rh, rw = (2, 2)
        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)
        out = input.view(batch_size, rh, rw, oc, h, w)
        out = out.permute(0, 3, 4, 1, 5, 2).contiguous()
        out = out.view(batch_size, oc, oh, ow)  # channel first
        return out
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder=nn.Sequential(
            ConvLayer(3,128),
            ConvLayer(128,256),
            ConvLayer(256,512),
            ConvLayer(512,1024),
            Flatten(),
            nn.Linear(1024*4*4,1024),
            nn.Linear(1024,1024*4*4),
            Reshape(),
            UpScale(1024,512),
        )
        self.decoder_a=nn.Sequential(
            UpScale(512,256),
            UpScale(256,128),
            UpScale(128,64),
            Conv2d(64, 3, kernel_size=5, padding=1),
            nn.Sigmoid(),
        )
        self.decoder_b=nn.Sequential(
            UpScale(512,256),
            UpScale(256,128),
            UpScale(128,64),
            Conv2d(64, 3, kernel_size=5, padding=1),
            nn.Sigmoid(),
        )
    def forward(self,x,select='A'):
        x=self.encoder(x)
        if select=='A':            
            x=self.decoder_a(x)
        else:
            x=self.decoder_b(x)
        return x
