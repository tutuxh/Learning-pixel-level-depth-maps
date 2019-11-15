# Learning Depth for Scene Reconstruction using Encoder-decoder Models

## Contents
0. [Introduction](#introduction)
0. [Quick Guide](#quick-guide)
0. [Models](#models)
0. [Results](#results)
0. [Citation](#citation)

## Introduction

This part contains pretrained models, which are stored in <a href="https://drive.google.com/file/d/1heAXjHVK0yQ4oKyR0qIyY4sRfSA_CapN/view?usp=sharing">Google Drive</a>.


The code and models verify the results in this paper "Learning Depth for Scene Reconstruction using Encoder-decoder Models".
The CNN models trained for depth estimation is available in this directory of results. The results from RGB images are the same as that in this paper when inputs are  the testing dataset in <a href="http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz">datasets</a>. The results from sparse depths and  RGBd data may be different from those in the paper slightly, due to the different depth samples. Additionally, the code provided can be used for inference on arbitrary images.

## Quick Guide
This code was tested with Python 3.6 and PyTorch 0.3.1.
Download the preprocessed <a href="http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz">NYU Depth V2</a> and <a href="http://datasets.lids.mit.edu/sparse-to-dense/data/kitti.tar.gz">KITTI Odometry dataset</a> that are provided by Fangchang Ma and used by the paper<a href="https://github.com/fangchangma/sparse-to-dense.pytorch"> "Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image"</a>.

## Models
A number of trained models can be downloaded <a href="https://drive.google.com/file/d/1heAXjHVK0yQ4oKyR0qIyY4sRfSA_CapN/view?usp=sharing">here</a> which are used to obtain the results reported in the paper on the benchmark datasets NYU-Depth-v2 and KITTI for indoor and outdoor scenes, respectively. 
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
## Results
The results are in this paper  "Learning Depth for Scene Reconstruction using Encoder-decoder Models". 
 
#### Citation
If you use our method or code in your work, please consider citing our paper.
The citation will be available after the paper is published.
