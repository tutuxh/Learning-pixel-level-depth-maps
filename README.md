# Learning Depth for Scene Reconstruction using Encoder-decoder Models

## Contents
0. [Introduction](#introduction)
0. [Quick Guide](#quick-guide)
0. [Depth estimation models](#models)
0. [The proposed SLAM system based on our depth estimation](#The-proposed-SLAM-system-based-on-our-depth-estimation)
0. [Results](#results)
0. [Citation](#citation)

## Introduction
This part includes pretrained models, which are stored in <a href="https://drive.google.com/file/d/1heAXjHVK0yQ4oKyR0qIyY4sRfSA_CapN/view?usp=sharing">Google Drive</a>.

The code and models verify the results in this paper "Learning Depth for Scene Reconstruction using Encoder-decoder Models".
The CNN models trained for depth estimation are available in this directory of results. The results are the same as that in the paper when inputs are RGB images on the testing dataset in <a href="http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz">datasets</a>. The results may be different from those in the paper slightly due to the different depth samples when inputs are sparse depths and RGBd data on the testing dataset in <a href="http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz">datasets</a>. Additionally, our provided code can be used for inference on arbitrary images.

## Quick Guide
This code was running with Python 3.6 and PyTorch 0.3.1.
We use the dataset <a href="https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html">NYU Depth V2</a> and <a href="http://www.cvlibs.net/datasets/kitti">KITTI Odometry dataset</a> to train and test our models.

## Depth estimation models
A number of trained models can be downloaded <a href="https://drive.google.com/file/d/1heAXjHVK0yQ4oKyR0qIyY4sRfSA_CapN/view?usp=sharing">here</a> which are used to acquire the results reported in the paper on the benchmark datasets NYU-Depth-v2 and KITTI for indoor and outdoor scenes, respectively. 
Run the main.py with options to obtain the prediction in this paper "Learning Depth for Scene Reconstruction using Encoder-decoder Models". For example, on the NYU-Depth-v2 dataset, the commands are:
```bash
python main.py -b 1 -m rgbd -s 20 --in_channels 512 --data [path_to_dataset] --epochs 30 --optimizer sgd --activation relu --dataset nyudepth --lr 0.01 --evaluate > log-rgbd-20-nyudepth-time.txt

python main.py -b 1 -m rgbd -s 50 --in_channels 512 --data [path_to_dataset] --epochs 30 --optimizer sgd --activation relu --dataset nyudepth --lr 0.01 --evaluate > log-rgbd-50-nyudepth-time.txt

python main.py -b 1 -m d -s 20 --in_channels 512 --data [path_to_dataset] --epochs 30 --optimizer sgd --activation relu --dataset nyudepth --lr 0.01 --evaluate > log-d-20-nyudepth-time.txt

python main.py -b 1 -m d -s 50 --in_channels 512 --data [path_to_dataset] --epochs 30 --optimizer sgd --activation relu --dataset nyudepth --lr 0.01 --evaluate > log-d-50-nyudepth-time.txt

python main.py -b 1 -m rgb -s 0 --in_channels 512 --data [path_to_dataset] --epochs 30 --optimizer sgd --activation relu --dataset nyudepth --lr 0.01 --evaluate > log-rgb-0-nyudepth-time.txt
```
On the KITTI dataset, the commands are:
```bash
python main.py -b 1 -m rgb -s 0 --in_channels 512 --data [path_to_dataset] --epochs 30 --optimizer sgd --activation relu --dataset kitti --lr 0.01 --evaluate > log-rgb-0-kitti-time.txt

python main.py -b 1 -m rgbd -s 50 --in_channels 512 --data [path_to_dataset] --epochs 30 --optimizer sgd --activation relu --dataset kitti --lr 0.01 --evaluate > log-rgbd-50-kitti-time.txt

python main.py -b 1 -m rgbd -s 100 --in_channels 512 --data [path_to_dataset] --epochs 30 --optimizer sgd --activation relu --dataset kitti --lr 0.01 --evaluate > log-rgbd-100-kitti-time.txt
```

## The proposed SLAM system based on our depth estimation
We compute the absolute trajectory error from the ground truth trajectory and the estimated trajectory.
For example, on the fr1_desk TUM sequence, the commands are:
```bash
python evaluate_ate.py groundtruth_fr1_desk.txt My_CameraTrajectory_fr1_desk.txt --verbose
```
On the fr3_str_tex_far TUM sequence, the commands are:
```bash
python evaluate_ate.py groundtruth_fr3_str_tex_far.txt My_CameraTrajectory_fr3_str_tex_far.txt --verbose
```

Then we can acquire our SLAM' absolute_translational_error.rmse, absolute_translational_error.mean, absolute_translational_error.median, absolute_translational_error.std, absolute_translational_error.min., and absolute_translational_error.max.

## Results
All results are in the paper "Learning Depth for Scene Reconstruction using Encoder-decoder Models".  

#### Citation
If you use our method or code in your work, please consider citing our paper.
The citation will be available after the paper is published.
