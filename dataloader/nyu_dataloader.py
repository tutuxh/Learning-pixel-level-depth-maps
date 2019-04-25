# -*-coding=utf-8-*-
import torch
import torch.utils.data as data
import h5py
import numpy as np
import os
from . import data_utils
from . import transforms

iheight, iwidth = 480, 640 # raw image size
oheight, owidth = 228, 304 # image size after pre-processing
to_tensor = transforms.ToTensor()

def val_transform(rgb, depth):
    depth_trans = depth

    # perform 1st part of data augmentation
    transform = transforms.Compose([
        # Resize
        transforms.Resize(240.0 / iheight),
        # CenterCrop
        transforms.CenterCrop((oheight, owidth)),
    ])
    depth_trans = transform(depth_trans)
    rgb_trans = transform(rgb)
    rgb_trans = np.asfarray(rgb_trans, dtype = 'float') / 255

    return rgb_trans, depth_trans

class NYUDataset(data.Dataset):
    h, w = oheight, owidth

    def __init__(self, data_list, type, modality, num_samples):

        self.data_list = data_list
        if type == 'val':
            self.transform = val_transform

        self.modality = modality
        self.num_samples = num_samples

    def __getitem__(self, index):
        h5_filename = self.data_list[index] # val目录下 目录+文件名
		# loader
        h5f = h5py.File(h5_filename, "r")
        rgb = np.array(h5f['rgb'])
        rgb = np.transpose(rgb, (1, 2, 0))
        depth = np.array(h5f['depth'])
		
        if self.transform is not None:
            rgb_trans, depth_trans = self.transform(rgb, depth)

        if self.modality == 'rgb':
            input_np = rgb_trans
            mask_keep = data_utils.create_relative_depth(depth_trans, self.num_samples) # num_samples ---relative loss
        elif self.modality == 'rgbd':
            input_np, mask_keep = data_utils.create_rgbd(rgb_trans, depth_trans, self.num_samples)
        elif self.modality == 'd':
            input_np, mask_keep = data_utils.create_sparse_depth(depth_trans, self.num_samples)

        input_tensor = to_tensor(input_np)
		
        if mask_keep is not None:
            mask_tensor = torch.from_numpy(mask_keep.astype(np.uint8).copy())
        else:
            mask_tensor = torch.Tensor([0])
			
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_trans)
        depth_tensor = depth_tensor.unsqueeze(0)
		
        h5f.close()
        return input_tensor, depth_tensor, mask_tensor, h5_filename.replace('/', '_')

    def __len__(self):
        return len(self.data_list)
