'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
import numpy as np
import models
import datasets
import math

from BatchAverage import BatchCriterion
from utils import *
from load_imagenet import imagenet, load_data

parser = argparse.ArgumentParser(description='PyTorch Seen Testing Category Training')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--log_dir', default='log/', type=str, 
                    help='log save path')
parser.add_argument('--model_dir', default='checkpoint/', type=str, 
                    help='model save path')
parser.add_argument('--test_epoch', default=1, type=int,
                    metavar='E', help='test every N epochs')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--batch-t', default=0.1, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--batch-m', default=1, type=float,
                    metavar='N', help='m for negative sum')
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--gpu', default='0,1,2,3', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')


parser.add_argument('--resnet', default='resnet18',  help='resnet18, resnet34, resnet50, resnet101')
parser.add_argument('--dataset', default='tinyImagenet',  help='[tinyImagenet]')
parser.add_argument('--trial', type=int, help='trial')
parser.add_argument('--adv', default=False, action='store_true', help='adversarial exmaple')
parser.add_argument('--eps', default=0.01, type=float, help='eps for adversarial')
parser.add_argument('--bn_adv_momentum', default=0.01, type=float, help='eps for adversarial')
parser.add_argument('--alpha', default=1.0, type=float, help='stregnth for regularization')
parser.add_argument('--debug', default=False, action='store_true', help='test_both_adv')
args = parser.parse_args() 

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def gen_adv(model, x, criterion, indexes):
    x_adv = Variable(x, requires_grad=True).to(device)
    adv_feat = model(x_adv, adv = True)
    feat = model(x, adv = True)
    features = torch.cat((feat,adv_feat), 0)
        
    tmp_loss = criterion(features, indexes)
    tmp_loss.backward()
    
    # generate adversarial example
    x_adv.data = x_adv.data + (args.eps * torch.sign(x_adv.grad.data))
    x_adv.grad.data.zero_()
    x_adv.requires_grad = False
    return x_adv


dataset = args.dataset
    
log_dir = args.log_dir + dataset + '_log/'
test_epoch = args.test_epoch
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)
if not os.path.isdir(args.model_dir + '/' + dataset):
    os.makedirs(args.model_dir + '/' + dataset)
     
suffix = args.dataset + '_{}_batch_{}_embed_dim_{}'.format(args.resnet, args.batch_size, args.low_dim)

if args.adv:
    suffix = suffix + '_adv_eps_{}_alpha_{}'.format(args.eps, args.alpha)
    suffix = suffix + '_bn_adv_momentum_{}_trial_{}'.format(args.bn_adv_momentum, args.trial)
else:
    suffix = suffix + '_trial_{}'.format(args.trial)

  
if len(args.resume)>0:
    suffix = suffix + '_r'

# log the output
test_log_file = open(log_dir + suffix + '.txt', "w")                
vis_log_dir = log_dir + suffix + '/'
if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data Preparation
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.2,1.)),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

if args.dataset == "tinyImagenet":
    root = '../datasets/tiny_imagenet.pickle'
else:
    raise NotImplementedError

trainset, _ = load_data(root)
trainset = imagenet(trainset, transform=transform_train) 
trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last =True)


ndata = trainset.__len__()

print('==> Building model..')
if args.adv:
    net = models.__dict__['MyResNet'](args.resnet, low_dim=args.low_dim, bn_adv_flag=True, bn_adv_momentum = args.bn_adv_momentum)
else:
    net = models.__dict__['MyResNet'](args.resnet, low_dim=args.low_dim, bn_adv_flag=False)

# define leminiscate: inner product within each mini-batch (Ours)

if device == 'cuda':
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


# define loss function: inner product loss within each mini-batch
criterion = BatchCriterion(args.batch_m, args.batch_t, args.batch_size)

net.to(device)
criterion.to(device)

   
# define optimizer
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed at 120, 160 and 200"""
    lr = args.lr
    if epoch >= 120 and epoch < 160:
        lr = args.lr * 0.1
    elif epoch >= 160 and epoch <200:
        lr = args.lr * 0.05
    elif epoch >= 200:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, (inputs1, inputs2, _, indexes) in enumerate(trainloader):
        data_time.update(time.time() - end)

        inputs1, inputs2, indexes = inputs1.to(device), inputs2.to(device), indexes.to(device)
        
        if args.adv:
            inputs_adv = gen_adv(net,inputs1, criterion, indexes)
        
        optimizer.zero_grad()
        inputs1_feat = net(inputs1)
        inputs2_feat = net(inputs2)
        features = torch.cat((inputs1_feat,inputs2_feat), 0)
        loss = criterion(features, indexes)
        
        if args.adv:
            adv_feat = net(inputs_adv,adv=True)
            loss += args.alpha*criterion(torch.cat((inputs1_feat, adv_feat), 0), indexes)
        
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(), 2*inputs1.size(0))         
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx%10 ==0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
                  epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))
        if args.debug:
            break 

best_acc = 0   
for epoch in range(start_epoch, start_epoch+100):
    
    # training 
    train(epoch)
    print('Finish epoch {}'.format(epoch), file = test_log_file)
    state = {
                'net': net.state_dict(),
                'epoch': epoch,
            }
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    torch.save(state, args.model_dir + '/' + dataset + '/' + suffix + '_best.t')

    test_log_file.flush()

    if args.debug:
        break
