import argparse
import os
import shutil
import sys
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim

import dataloader.data_utils as data_utils
from dataloader.data_utils import get_dataloader

from metrics import AverageMeter, Result
import utils

from models.dpn.dpnet import DualPathNet
from models.upsampling import DeConv #Deconv_Block_3X3, Deconv_Block_2X2
# UpConvBlock FastUpConvBlock UpProjectBlock FastUpProjectBlock

modality_names = data_utils.modality_names # modality_names = ['rgb', 'rgbd', 'd'] # , 'g', 'gd'
data_sets = data_utils.data_sets # data_sets = ['nyudepth', 'make3d', 'kitti']

model_names = DualPathNet.arch_names 
#loss_names = ['l1', 'l2', 'torchl1', 'huber', 'relative']
decoder_names = DeConv.names + ['fast_upproj', 'fast_upconv', 'upproj', 'upconv']

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Depth Estimation')

parser.add_argument('--data', metavar='DATA', default='nyu_data',
                    help='dataset directory: (nyudepthv2/kitti/make3d)')
parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
                    choices=modality_names,
                    help='modality: ' +
                        ' | '.join(modality_names) +
                        ' (default: rgb)')
parser.add_argument('-s', '--num-samples', default=0, type=int, metavar='N',
                    help='number of sparse depth samples (default: 0)')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    default=True, help='use ImageNet pre-trained weights (default: True)')
parser.add_argument('--in_channels', type=int, required=True, help='the decoder in_channels')
parser.add_argument('--optimizer', default='sgd', type=str, required=True, help='optimizer option')
parser.add_argument('--activation', default='relu', type=str, required=True, help='activation option')
parser.add_argument('--dataset', default='nyudepth',choices=data_sets, type=str, required=True, help='datasets option')	
	
fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae', 'delta1', 'delta2', 'delta3']
best_result = Result()
best_result.set_to_worst()

def main():
    global args, best_result, output_directory, test_csv
    args = parser.parse_args()
    if not args.data.startswith('/'):
        args.data = os.path.join('../', args.data)
    
    # create results folder, if not already exists
    output_directory = os.path.join('results',
        'Dataset={}.modality={}.nsample={}.bs={}.channels={}.optimizer={}'.
        format(args.dataset, args.modality, args.num_samples, args.batch_size, args.in_channels, args.optimizer))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')
    
    # Data loading code
    print("=> creating data loaders ...")
	
    val_on, val_loader, h, w = get_dataloader(args.dataset, args.data, args.batch_size, args.modality, args.num_samples, args.workers)
    val_len = len(val_loader)
    out_size = h, w
    print(out_size)
    print("test dataloader len={}".format(val_len))
	
    print("=> data loaders created.")
	
    # evaluation mode
    if args.evaluate:
        best_model_filename = os.path.join(output_directory, 'model_best.pth.tar')
        if os.path.isfile(best_model_filename):
            print("=> loading previous model '{}'".format(best_model_filename))
            checkpoint = torch.load(best_model_filename)
            epoch = checkpoint['epoch']
            best_result = checkpoint['best_result']
            model = checkpoint['model']
            print("=> loaded previous model (epoch={})".format(checkpoint['epoch']))
            print("\n(\n rmse={:.5f},\n absrel={:.5f}, \n delta1={:.5f},\n delta2={:.5f}, \n delta3={:.5f}\n)".format(best_result.rmse, best_result.absrel, best_result.delta1, best_result.delta2, best_result.delta3))
        else:
            print("=> no previous model found at '{}'".format(best_model_filename))
        evaluate(val_loader, model, checkpoint['epoch'], write_to_file=True)
        #validate(val_loader, model, checkpoint['epoch'], write_to_file=False)
        return

    # create new model
    else:
        print("please Please specify the evaluation mode: --evaluate ")

    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    print(model)
    print("=> model transferred to GPU.")

def evaluate(val_loader, model, epoch, write_to_file=True):
    #print(model)
    average_meter = AverageMeter()
    model.eval()    # switch to evaluate mode
    for i, (input, target, mask, h5name) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        depth_pred = model(input_var)
        result = Result()
        output1 = torch.index_select(depth_pred.data, 1, torch.cuda.LongTensor([0]))
        result.evaluate(output1, target)
        average_meter.update(result, input.size(0))
        
        if args.modality == 'd':
            img_merge = None
        else:
            if args.modality == 'rgb':
                rgb = input
            elif args.modality == 'rgbd':
                rgb = input[:, :3, :, :]
				
                sparse_d = input[:, -1, :, :]
                sparse_depth_np = utils.depth_to_np(sparse_d)
                sd_filename = output_directory + '/' + str(epoch)+'_'+ '_' + h5name[0] + '_s_depth.png'
                utils.save_image(sparse_depth_np, sd_filename)

            rgb_np, depth_np, pred_np = utils.image_3_row(rgb, target, depth_pred)
            rgb_filename = output_directory + '/' + str(epoch)+'_'+  '_' + h5name[0] + '_rgb.png'
            utils.save_image(rgb_np, rgb_filename)
            depth_filename = output_directory + '/' + str(epoch)+'_'+  '_' + h5name[0] + '_depth.png'
            utils.save_image(depth_np, depth_filename)
            pred_filename = output_directory + '/' + str(epoch)+'_'+ '_' + h5name[0] + '_pred.png'
            utils.save_image(pred_np, pred_filename)

    avg = average_meter.average()

    print('\n\nnew prediction with data:\n*\n'
        'RMSE={average.rmse:.5f}\n'
        'MAE={average.mae:.5f}\n'
        'Delta1={average.delta1:.5f}\n'
        'Delta2={average.delta2:.5f}\n'
        'Delta3={average.delta3:.5f}\n'
        'REL={average.absrel:.5f}\n'
        'Lg10={average.lg10:.5f}\n'.format(
        average=avg))

    if write_to_file:
        with open(test_csv, 'a') as csvfile: 
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3})

if __name__ == '__main__':
    main()

