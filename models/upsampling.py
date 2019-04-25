# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

import math
import collections
import numpy as np

###########deconvlution#########################
class Deconv_Block_3X3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Deconv_Block_3X3, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride=(2, 2), padding = (1, 1), output_padding = (1, 1), bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Deconv_Block_2X2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Deconv_Block_2X2, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, (2, 2), stride = (2, 2), bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Decoder(nn.Module):
    # Decoder is the base class for all decoders
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class DeConv(Decoder):
    names = ['deconv{}'.format(i) for i in range(2, 10)]
    def __init__(self, in_channels, kernel_size):
        assert kernel_size >= 2, "kernel_size out of range: {}".format(kernel_size)
        super(DeConv, self).__init__()
        self.layer1 = self.convt(in_channels, kernel_size)
        self.layer2 = self.convt(in_channels // 2, kernel_size)
        self.layer3 = self.convt(in_channels // (2 ** 2), kernel_size)
        self.layer4 = self.convt(in_channels // (2 ** 3), kernel_size)

    def convt(self, in_channels, kernel_size):
        stride = 2
        padding = (kernel_size - 1) // 2
        output_padding = kernel_size % 2

        assert -2 - 2 * padding + kernel_size + output_padding == 0, "deconv parameters incorrect"
		
        module_name = "deconv{}".format(kernel_size)
        return nn.Sequential(collections.OrderedDict([
			  (module_name, nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size,
					stride, padding, output_padding, bias = False)),
			  ('batchnorm', nn.BatchNorm2d(in_channels//2)),
			  ('relu',      nn.ReLU(inplace = True)),
			]))

def choose_decoder(decoder_name, in_channels):
    #iheight, iwidth = 10, 8
    # UpConvBlock FastUpConvBlock UpProjectBlock FastUpProjectBlock
    #'deconv[2-10]', ['fast_upproj', 'fast_upconv', 'upproj', 'upconv']
    if decoder_name in DeConv.names: # deconv[2-10]
        assert decoder_name[:6] == 'deconv'
        assert len(decoder_name) == 7 
        kernel_size = int(decoder_name[6])
        decoder = DeConv(in_channels, kernel_size)
    elif decoder_name == 'upconv':
	    decoder = UpSampling_Decoder(UpConvBlock, in_channels)
    elif decoder_name == 'fast_upconv':
	    decoder = UpSampling_Decoder(FastUpConvBlock, in_channels)
    elif decoder_name == 'upproj':
	    decoder = UpSampling_Decoder(UpProjectBlock, in_channels)
    elif decoder_name == 'fast_upproj':
	    decoder = UpSampling_Decoder(FastUpProjectBlock, in_channels)
    return decoder
	
#############################################################
def interleave(tensors, dim): #dim=2,dim=3
    s1, s2, s3, s4 = tensors[0].size()
    m = [s1, s2, s3, s4]
    m[dim] = m[dim] * 2
    return torch.stack(tensors, dim = dim + 1).view(m).contiguous()

#https://discuss.pytorch.org/t/how-to-use-maxunpool2d-without-indices/11331/2
#double the size of feature maps
def unpooling(input):
    n = input.size()[0]
    c = input.size()[1]
    h = input.size()[2]
    w = input.size()[3]

    unpool = nn.MaxUnpool2d(2, stride = 2)

    #2*2 => 4*4
    list = []
    lh, lw = 2 * h, 2 * w
    for i in range(1, lh, 2):
        for j in range(0, lw, 2):
            list.append(j + (i - 1) * lw)

    list_np = np.array(list)
    list_np = list_np.reshape(h, w)

    all_list = np.zeros((n, c, h, w))
    all_list[:] = list_np
    indices = Variable(torch.from_numpy(all_list).long()).cuda()

    output = unpool(input, indices)
    return output

#Deeper Depth Prediction with Fully Convolutional Residual Networks, 3DV2016
# up-projection
class UpProjectBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpProjectBlock, self).__init__()
        kW, kH = 5, 5
        dW, dH = 1, 1
        padW = 2 #(kW - 1)/2
        padH = 2 #(kH - 1)/2

        self.conv1 = nn.Conv2d(in_channels, out_channels, (kW, kH), (dW, dH), (padW, padH), bias = False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1) , (1, 1), bias = False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kW, kH), (dW, dH), (padW, padH), bias = False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu1 = nn.ReLU(inplace = True)
        self.relu2 = nn.ReLU(inplace = True)

    def forward(self, x, BN = True):
        x = unpooling(x)
		
        out1 = self.conv1(x)
        if BN:
            out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        out1 = self.conv2(out1)
        if BN:
            out1 = self.bn2(out1)
		
        out2 = self.conv3(x)
        if BN:
            out2 = self.bn3(out2)
		
        out = out1 + out2
        return self.relu2(out)

class FastUpProjectBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FastUpProjectBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), bias = False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (2, 3), bias = False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (3, 2), bias = False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, (2, 2), bias = False)

        self.conv5 = nn.Conv2d(in_channels, out_channels, (3, 3), bias = False)
        self.conv6 = nn.Conv2d(in_channels, out_channels, (2, 3), bias = False)
        self.conv7 = nn.Conv2d(in_channels, out_channels, (3, 2), bias = False)
        self.conv8 = nn.Conv2d(in_channels, out_channels, (2, 2), bias = False)

        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv9 = nn.Conv2d(out_channels, out_channels, 3, padding = 1, bias = False)

    def forward(self, x, BN = True):
        out1 = self.unpool_as_conv(x, BN = BN, ReLU = True, id = 1)
        out1 = self.conv9(out1)

        if BN:
            out1 = self.bn2(out1)

        out2 = self.unpool_as_conv(x, BN = BN, ReLU = False, id = 2)
        out = out1 + out2
        out = self.relu(out)
        return out

    def unpool_as_conv(self, x, BN = True, ReLU = True, id = 1):
        if id==1:
            out1 = self.conv1(F.pad(x, (1, 1, 1, 1)))
            out2 = self.conv2(F.pad(x, (1, 1, 0, 1)))
            out3 = self.conv3(F.pad(x, (0, 1, 1, 1)))
            out4 = self.conv4(F.pad(x, (0, 1, 0, 1)))
#            if BN:
#                out1 = self.bn1(out1)
#                out2 = self.bn2(out2)
#                out3 = self.bn3(out3)
#                out4 = self.bn4(out4)
        else:
            out1 = self.conv5(F.pad(x, (1, 1, 1, 1)))
            out2 = self.conv6(F.pad(x, (1, 1, 0, 1)))
            out3 = self.conv7(F.pad(x, (0, 1, 1, 1)))
            out4 = self.conv8(F.pad(x, (0, 1, 0, 1)))
#            if BN:
#                out1 = self.bn5(out1)
#                out2 = self.bn6(out2)
#                out3 = self.bn7(out3)
#                out4 = self.bn8(out4)

        # dim=2 for h,  dim=3 for w
        ac_t = interleave([out1, out2], dim = 2)
        bd_t = interleave([out3, out4], dim = 2)
        Y = interleave([ac_t, bd_t], dim = 3)

        if id == 1:
            if BN:
                Y = self.bn1_1(Y)
        else:
            if BN:
                Y = self.bn1_2(Y)

        if ReLU:
            Y = self.relu(Y)

        return Y
		
# up-Convolution
#each output pixel has a reception field of 3×3
class UpConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        #5×5 convolution, reduce the number of channels to half
        kW, kH = 5, 5
        dW, dH = 1, 1
        padW = 2 #(kW - 1)/2
        padH = 2 #(kH - 1)/2
        self.conv = nn.Conv2d(in_channels, out_channels, (kW, kH), (dW, dH), (padW, padH), bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x, BN = True):
        x = unpooling(x)
        x = self.conv(x)
        if BN:
            x = self.bn(x)
        x = self.relu(x)
        return x
	
class FastUpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FastUpConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), bias = False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (2, 3), bias = False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (3, 2), bias = False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, (2, 2), bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)      
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x, BN = True):
        out1 = self.conv1(F.pad(x, (1, 1, 1, 1)))
        out2 = self.conv2(F.pad(x, (1, 1, 0, 1)))
        out3 = self.conv3(F.pad(x, (0, 1, 1, 1)))
        out4 = self.conv4(F.pad(x, (0, 1, 0, 1)))

        if BN:
            out1 = self.bn1(out1)
            out2 = self.bn2(out2)
            out3 = self.bn3(out3)
            out4 = self.bn4(out4)
			
        ac_t = interleave([out1, out2], dim = 2)
        bd_t = interleave([out3, out4], dim = 2)

        Y = interleave([ac_t, bd_t], dim = 3)
        Y = self.relu(Y)
        return Y

class UpSampling_Decoder(Decoder):
    def __init__(self, block, in_channels):
        super(UpSampling_Decoder, self).__init__()
        self.layer1 = self.make_deconv_layer(block, in_channels, in_channels // 2)
        self.layer2 = self.make_deconv_layer(block, in_channels// 2, in_channels // (2 ** 2))
        self.layer3 = self.make_deconv_layer(block, in_channels // (2 ** 2), in_channels // (2 ** 3))
        self.layer4 = self.make_deconv_layer(block, in_channels // (2 ** 3), in_channels // (2 ** 4))
		
    def make_deconv_layer(self, block, in_channels, out_channels):
        return block(in_channels, out_channels)
			
def init_weights(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d): 
        m.weight.data.fill_(1)
        if m.bias is not None: 
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.fill_(1)
        if m.bias is not None: 
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
	
if __name__ == '__main__':
    x = Variable(torch.Tensor([[[[ 1,  2,  3,  4],[ 5,  6,  7,  8],[ 9, 10, 11, 12], [13, 14, 15, 16]]]])) # Variable(torch.arange(1,17).view(4,4).unsqueeze(0).unsqueeze(0))
    #x = Variable(torch.rand(1, 2, 4, 4))
    bn = False
    print(x) 
	# UpConvBlock
    upconv = UpConvBlock(1, 2)
    upconv.apply(init_weights)
    print("----------------weight-----------")
    print(upconv.conv.weight.data)
    upconv_out = upconv(x, BN = bn)
    print(upconv_out.size())
    print(upconv_out)
	# FastUpConvBlock
    fast_upconv = FastUpConvBlock(1, 2)
    fast_upconv.apply(init_weights)
    fast_upconv_out = fast_upconv(x, BN = bn)
    print(fast_upconv_out.size())
    print(fast_upconv_out)
	
    # UpProjectBlock
    upproj = UpProjectBlock(1, 2)
    upproj.apply(init_weights)
    upproj_out = upproj(x, BN = bn)
    print(upproj_out.size())
    print(upproj_out)
	
	# FastUpProjectBlock
    fast_upproj = FastUpProjectBlock(1, 2)
    fast_upproj.apply(init_weights)
    fast_upproj_out = fast_upproj(x, BN = bn)
    print(fast_upproj_out.size())
    print(fast_upproj_out)