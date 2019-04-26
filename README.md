# Learning Pixel-level Depth Maps by using an Encoder-decoder Model

## Contents
0. [Introduction](#introduction)
0. [Quick Guide](#quick-guide)
0. [Models](#models)
0. [Results](#results)
0. [Citation](#citation)

## Introduction

This part contains pretrained models, which are stored in <a href="https://drive.google.com/file/d/1heAXjHVK0yQ4oKyR0qIyY4sRfSA_CapN/view?usp=sharing">Google Drive</a>.


The code and models verify the results in this paper "Learning Pixel-level Depth Maps by using an Encoder-decoder Model".
The CNN models trained for depth estimation is available in this directory of results.   The models can be downloaded <a href="https://drive.google.com/file/d/1heAXjHVK0yQ4oKyR0qIyY4sRfSA_CapN/view?usp=sharing">here</a>, that are used to obtain the results reported in the paper on the benchmark datasets NYU-Depth-v2 and KITTI for indoor and outdoor scenes, respectively.   The results from RGB images  are the same as that in this paper when inputs are  the testing dataset  in <a href="http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
">datasets</a>. The results from sparse depths and  RGBd data may be different from those in the paper slightly, due to the different depth samples. Additionally, the code provided can be used for inference on arbitrary images.

