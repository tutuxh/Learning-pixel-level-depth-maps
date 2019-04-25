# -*- coding: UTF-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from .dpn import *
from ..utils import weights_init
from ..upsampling import *

class DualPathNet(nn.Module):
    arch_names = ['dpn68', 'dpn68b', 'dpn92', 'dpn92b', 'dpn98', 'dpn107', 'dpn131']
	
    def __init__(self, arch, decoder_name, input_channels, in_channels, out_size):
        super(DualPathNet, self).__init__()
		
        arch_out = 2688
        if arch == 'dpn68':
            original_model = dpn68(pretrained=True, input_channels = input_channels) # 832
            arch_out = 832
        elif arch == 'dpn68b':
            original_model = dpn68b(pretrained=True, input_channels = input_channels)  # 832
            arch_out = 832
        elif arch == 'dpn92': 
            original_model = dpn92(pretrained = True, extra = False, input_channels = input_channels) # 2688
        elif arch == 'dpn92b': 
            original_model = dpn92(pretrained = True, extra = True, input_channels = input_channels) # 2688, 8, 10
        elif arch == 'dpn107': 
            original_model = dpn107(pretrained=True, input_channels = input_channels)
        elif arch == 'dpn98': 
            original_model = dpn98(pretrained=True, input_channels = input_channels)
        elif arch == 'dpn131': 
            original_model = dpn131(pretrained=True, input_channels = input_channels)

        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.conv2 = nn.Conv2d(arch_out, in_channels, 1, bias = False) 
		# 1344 672 336 168 84 42
        self.bn2 = nn.BatchNorm2d(in_channels)# 1024 512 256 128 64
		
        # UpConvBlock FastUpConvBlock UpProjectBlock FastUpProjectBlock
        #'deconv[2-10]', ['fast_upproj', 'fast_upconv', 'upproj', 'upconv']
        self.decoder = choose_decoder(decoder_name, in_channels)
        """
        self.conv3 = nn.Conv2d(in_channels // (2 ** 4), 1, 3, padding = 1, bias = False) # 42
        self.upsample = nn.Upsample(size = (out_size[0], out_size[1]), mode = 'bilinear') #nn.UpsamplingBilinear2d(size = (480, 640))
        
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)
        """
        self.conv3_1 = nn.Conv2d(in_channels // 16, in_channels // 32, 7, padding = 1, bias = False)
        self.bn3_1 = nn.BatchNorm2d(in_channels // 32)
        self.relu3_1 = nn.ReLU(inplace = True)
        self.conv3_2 = nn.Conv2d(in_channels // 32, in_channels // 64, 7, padding = 1, bias = False)
        self.bn3_2 = nn.BatchNorm2d(in_channels // 64)
        self.relu3_2 = nn.ReLU(inplace = True)
        self.conv3_3 = nn.Conv2d(in_channels // 64, in_channels // 128, (7, 1), padding = 0, bias = False)
        self.bn3_3 = nn.BatchNorm2d(in_channels // 128)
        self.relu3_3 = nn.ReLU(inplace = True)
		
        #self.conv3 = nn.Conv2d(in_channels // (2 ** 4), 1, 3, padding = 1, bias = False)	
        self.upsample = nn.ConvTranspose2d(in_channels // 128, 1, (4, 4), stride=(2, 2), padding=(1, 1), bias = False)

		# nyu_depth=(228, 304)
        #self.upsample = nn.Upsample(size = (out_size[0], out_size[1]), mode = 'bilinear') #nn.UpsamplingBilinear2d(size = (480, 640))
        
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3_1.apply(weights_init)
        self.conv3_2.apply(weights_init)
        self.conv3_3.apply(weights_init)
        self.bn3_1.apply(weights_init)
        self.bn3_2.apply(weights_init)
        self.bn3_3.apply(weights_init)
        self.upsample.apply(weights_init)
		
    def forward(self, x):
        x = self.features(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.decoder(x)
        #x = self.conv3(x)
        #x = self.upsample(x)
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = self.relu3_3(x)
        x = self.upsample(x)
        return x
