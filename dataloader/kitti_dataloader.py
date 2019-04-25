# -*-coding=utf-8-*-
import torch
import torch.utils.data as data
import h5py
import numpy as np
import os
from . import data_utils
from . import transforms

iheight, iwidth = 376, 1241 # raw image size
oheight, owidth = 228, 912 # image size after pre-processing
to_tensor = transforms.ToTensor()

meanstdRGB = {
   "mean" : np.array([ 0.485, 0.456, 0.406 ]),
   "std" : np.array([ 0.229, 0.224, 0.225 ])
}

def val_transform(rgb, depth):
    rgb = np.asfarray(rgb, dtype = 'float')
    # perform 1st part of data augmentation
    rgb_transform = transforms.Compose([
		transforms.Crop(130, 370, 10, 1210),
		transforms.ColorNormalize(meanstdRGB),
        transforms.CenterCrop((oheight, owidth)),
    ])
    depth_transform = transforms.Compose([
		transforms.Crop(130, 370, 10, 1210),
        transforms.CenterCrop((oheight, owidth)),
    ])
	
    rgb_np = rgb_transform(rgb)
    rgb_np = np.asfarray(rgb_np, dtype = 'float') / 255
    depth_np = depth_transform(depth)

    return rgb_np, depth_np

class KITTIDataset(data.Dataset):
    h, w = oheight, owidth
    def __init__(self, data_list, type, modality, num_samples):

        self.data_list = data_list
        if type == 'val':
            self.transform = val_transform

        self.modality = modality
        self.num_samples = num_samples
		
    def __getitem__(self, index):
        h5_filename = self.data_list[index] # train目录下 目录+文件名
		# loader
        h5f = h5py.File(h5_filename, "r")
        rgb = np.array(h5f['rgb'])
        rgb = np.transpose(rgb, (1, 2, 0))
        depth = np.array(h5f['depth']).astype(np.float32) # float16->float32
        #print(rgb.shape, depth.shape, rgb.dtype, depth.dtype)
        if self.transform is not None:
            rgb_np, depth_np = self.transform(rgb, depth)

        # color normalization
        # rgb_tensor = normalize_rgb(rgb_tensor)
        # rgb_np = normalize_np(rgb_np)
        
        if self.modality == 'rgb':
            input_np = rgb_np
            mask_keep = data_utils.create_relative_depth(depth_np, self.num_samples) # num_samples ---relative loss
        elif self.modality == 'rgbd':
            input_np, mask_keep = data_utils.create_rgbd(rgb_np, depth_np, self.num_samples)
        elif self.modality == 'd':
            input_np, mask_keep = data_utils.create_sparse_depth(depth_np, self.num_samples)

        input_tensor = to_tensor(input_np)
		
        if mask_keep is not None:
            mask_tensor = torch.from_numpy(mask_keep.astype(np.uint8).copy())
        else:
            mask_tensor = torch.Tensor([0])
			
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)
		
        h5f.close()
        #print(input_tensor.size(), depth_tensor.size(), mask_tensor.size())
        return input_tensor, depth_tensor, mask_tensor, h5_filename.replace('/', '_')

    def __len__(self):
        return len(self.data_list)
