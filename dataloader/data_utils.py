# -*-coding=utf-8-*-
import torch
import torch.utils.data as data_utils
import h5py
import numpy as np
import random
import os
from .nyu_dataloader import NYUDataset
from .kitti_dataloader import KITTIDataset

modality_names = ['rgb', 'rgbd', 'd'] # , 'g', 'gd'
data_sets = ['nyudepth', 'kitti']

def get_prob(depth, num_samples):
    nValidPixels = np.sum(np.greater(depth, 0)) #torch.sum(torch.gt(depth, 0))
    if nValidPixels == 0:
        return 0
    prob = float(num_samples) / nValidPixels
    prob = prob < 1 and prob or 1
    return prob
	
# create_relative_depth create_sparse_depth create_rgbd
def create_relative_depth(depth, num_samples):
    if num_samples == 0:
        return None
		
    prob = get_prob(depth, num_samples)
    #prob = float(num_samples) / depth.size
    mask_keep = np.random.uniform(0, 1, depth.shape) < prob
    #sparse_depth = np.zeros(depth.shape)
    #sparse_depth[mask_keep] = depth[mask_keep]
    return mask_keep
	
def create_sparse_depth(depth, num_samples):
    prob = get_prob(depth, num_samples)
    #prob = float(num_samples) / depth.size
    mask_keep = np.random.uniform(0, 1, depth.shape) < prob
    sparse_depth = np.zeros(depth.shape)
    sparse_depth[mask_keep] = depth[mask_keep]
    return sparse_depth, mask_keep

def create_rgbd(rgb, depth, num_samples):
    sparse_depth, mask_keep = create_sparse_depth(depth, num_samples)
    # rgbd = np.dstack((rgb[:,:,0], rgb[:,:,1], rgb[:,:,2], sparse_depth))
    rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
    return rgbd, mask_keep

#########################################	
def read_list(filename):
    f = open(filename, 'r')
    data = f.read().strip('\n')
    f.close()
    data_list = data.split('\n')
    return data_list

# args.data, args.batch_size, args.modality, args.num_samples, args.workers	
def get_dataloader(data_set, data_dir, batch_size, modality, num_samples, workers):

    val_file = os.path.join(data_dir, 'val.txt')
    val_list = read_list(val_file)

    if data_set == 'nyudepth':
        val_dataset = NYUDataset(val_list, type='val', modality = modality, num_samples = num_samples)
        h, w = NYUDataset.h, NYUDataset.w
    elif data_set == 'kitti':
        #val_list = random.sample(val_list, 3200) # random get 3200 validataion image
        val_dataset = KITTIDataset(val_list, type='val', modality = modality, num_samples = num_samples)
        h, w = KITTIDataset.h, KITTIDataset.w

    val_loader = data_utils.DataLoader(val_dataset, 1, shuffle = False, num_workers = workers, drop_last = False)
    return len(val_list), val_loader, h, w
