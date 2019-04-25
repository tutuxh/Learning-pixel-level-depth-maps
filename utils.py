import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

cmap = plt.cm.jet

def merge_into_row(input, target, depth_pred):
    # H, W, C
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0))
    depth = np.squeeze(target.cpu().numpy())
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    # H, W, C
    depth = 255 * cmap(depth)[:,:,:3] 
    pred = np.squeeze(depth_pred.data.cpu().numpy())
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    # H, W, C
    pred = 255 * cmap(pred)[:,:,:3] 
    merge_img = np.hstack([rgb, depth, pred])
    
    # merge_img.save(output_directory + '/comparison_' + str(epoch) + '.png')
    return merge_img

def depth_to_np(sparse_d):
    depth = np.squeeze(sparse_d.cpu().numpy())
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:,:,:3] # H, W, C
    return depth
	
def image_3_row(input, target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth = np.squeeze(target.cpu().numpy())
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:,:,:3] # H, W, C
    pred = np.squeeze(depth_pred.data.cpu().numpy())
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    pred = 255 * cmap(pred)[:,:,:3] # H, W, C
    return rgb, depth, pred
	
def add_row(merge_img, row):
    return np.vstack([merge_img, row])

def save_image(merge_img, file_name):
    merge_img = Image.fromarray(merge_img.astype('uint8'))
    merge_img.save(file_name)

