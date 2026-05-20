import os
import time
import argparse
import random
import math
from importlib import reload, import_module

from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data
from dataset.ImageNetMask import imagenet_r_mask

import torch    
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import timm
import numpy as np

import tta_library.tent as tent
import tta_library.sar as sar
import tta_library.cotta as cotta
import tta_library.foa_bp as foa_bp

from tta_library.sam import SAM
from tta_library.t3a import T3A
from tta_library.foa import FOA
from tta_library.foa_shift import Shift
from tta_library.lame import LAME

from calibration_library.metrics import ECELoss

from quant_library.quant_utils.models import get_net
from quant_library.quant_utils import net_wrap
import quant_library.quant_utils.datasets as datasets
from quant_library.quant_utils.quant_calib import HessianQuantCalibrator

from models.vpt import PromptViT

def validate_adapt(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    
    outputs_list, targets_list = [], []
    with torch.no_grad():
        end = time.time()
        for i, dl in enumerate(val_loader):
            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            output = model(images)
            
            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            outputs_list.append(output)
            targets_list.append(target)
            
            if i % args.print_freq == 0:
                progress.display(i)
                
    # Calculate ECE
    outputs_all = torch.cat(outputs_list, dim=0)
    targets_all = torch.cat(targets_list, dim=0)
    
    ece_loss = ECELoss(n_bins=15)
    ece = ece_loss(outputs_all, targets_all)
    
    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} ECE {ece:.3f}')
    
    # Save results
    results_path = os.path.join(args.output, f"{args.algorithm}_{args.tag}_results.txt")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        f.write(f"Accuracy@1: {top1.avg:.3f}\n")
        f.write(f"Accuracy@5: {top5.avg:.3f}\n")
        f.write(f"ECE: {ece:.3f}\n")
        f.write(f"Dataset: {args.data_corruption}\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Quantization level: {args.quantization_level}\n")
    
    return top1.avg, ece

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main():
    parser = argparse.ArgumentParser(description='Test Time Adaptation')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--data', default='/mnt/cephfs/dataset/TTA/imagenet', type=str, help='path to imagenet')
    parser.add_argument('--data_corruption', default='/mnt/cephfs/dataset/TTA/imagenet-c', type=str, help='path to imagenet-c')
    parser.add_argument('--output', default='./outputs', type=str, help='output directory')
    parser.add_argument('--algorithm', default='foa', type=str, help='algorithm to use')
    parser.add_argument('--tag', default='', type=str, help='tag for output')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
    parser.add_argument('--num_prompts', default=3, type=int, help='number of prompts')
    parser.add_argument('--fitness_lambda', default=0.4, type=float, help='fitness lambda for FOA')
    parser.add_argument('--quantization_level', default=8, type=int, help='quantization level (8 or 6)')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    # Load model
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    
    # Wrap model with prompts
    model = PromptViT(model, num_prompts=args.num_prompts)
    
    # Apply quantization if specified
    if args.quantization_level == 8:
        model = net_wrap.quantize_model(model, bits=8)
    elif args.quantization_level == 6:
        model = net_wrap.quantize_model(model, bits=6)
    
    # Move to GPU
    model = model.cuda()
    
    # Initialize adaptation method
    if args.algorithm == 'foa':
        # Load source domain statistics from ImageNet validation set
        train_dataset = datasets.ImageNetLoaderGenerator(
            root=args.data, 
            dataset_name='imagenet', 
            train_batch_size=args.batch_size, 
            test_batch_size=args.batch_size
        )
        
        # Create data loader for source statistics
        train_loader = train_dataset.train_loader()
        
        # Initialize FOA
        foa = FOA(model, fitness_lambda=args.fitness_lambda)
        foa.obtain_origin_stat(train_loader)
        
        # Use FOA as the model for adaptation
        model = foa
        
    elif args.algorithm == 'tent':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model = tent.Tent(model, optimizer, steps=1, episodic=False)
    
    # Load test data
    test_dataset = datasets.ImageNetLoaderGenerator(
        root=args.data_corruption, 
        dataset_name='imagenet-c', 
        train_batch_size=args.batch_size, 
        test_batch_size=args.batch_size
    )
    
    val_loader = test_dataset.test_loader()
    
    # Validate
    acc1, ece = validate_adapt(val_loader, model, args)
    
    print(f"Final results - Acc@1: {acc1:.3f}, ECE: {ece:.3f}")

if __name__ == '__main__':
    main()