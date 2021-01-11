import os
import torch
import torchvision
import argparse

from torch.autograd import Variable
import numpy as np
from model import load_model, save_model
from modules import NT_Xent
from modules.transformations import TransformsSimCLR
from modules.transformations import TransformsSimCLR_imagenet
from utils import mask_correlated_samples
from load_imagenet import imagenet, load_data

parser = argparse.ArgumentParser(description='PyTorch Seen Testing Category Training')
parser.add_argument('--batch_size', default=256, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--workers', default=4, type=int, help='workers')
parser.add_argument('--epochs', default=100, type=int,help='epochs')
parser.add_argument('--resnet', default="resnet18", type=str, help="resnet")
parser.add_argument('--normalize', default=True, action='store_true', help='normalize')
parser.add_argument('--projection_dim', default=64, type=int,help='projection_dim')
parser.add_argument('--optimizer', default="Adam", type=str, help="optimizer")
parser.add_argument('--weight_decay', default=1.0e-6, type=float, help='weight_decay')
parser.add_argument('--temperature', default=0.5, type=float, help='temperature')
parser.add_argument('--model_path', default='log/', type=str, 
                    help='model save path')
parser.add_argument('--model_dir', default='checkpoint/', type=str, 
                    help='model save path')


parser.add_argument('--dataset', default='CIFAR10',  
                    help='[CIFAR10, CIFAR100, tinyImagenet]')
parser.add_argument('--gpu', default='0', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--trial', type=int, help='trial')
parser.add_argument('--adv', default=False, action='store_true', help='adversarial exmaple')
parser.add_argument('--eps', default=0.01, type=float, help='eps for adversarial')
parser.add_argument('--bn_adv_momentum', default=0.01, type=float, help='batch norm momentum for advprop')
parser.add_argument('--alpha', default=1.0, type=float, help='weight for contrastive loss with adversarial example')
parser.add_argument('--debug', default=False, action='store_true', help='debug mode')

args = parser.parse_args() 

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def gen_adv(model, x_i, criterion):
    x_i = x_i.detach()
    h_i, z_i = model(x_i, adv=True)
    
    x_j_adv = Variable(x_i, requires_grad=True).to(args.device)
    h_j_adv, z_j_adv = model(x_j_adv, adv=True)
    tmp_loss = criterion(z_i, z_j_adv)
    tmp_loss.backward()
    x_j_adv.data = x_j_adv.data + (args.eps * torch.sign(x_j_adv.grad.data))
    x_j_adv.grad.data.zero_()
    
    x_j_adv.detach()
    x_j_adv.requires_grad = False
    return x_j_adv

def train(args, train_loader, model, criterion, optimizer):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):

        optimizer.zero_grad()
        x_i = x_i.to(args.device)
        x_j = x_j.to(args.device)

        # positive pair, with encoding
        h_i, z_i = model(x_i)
        ori_data = x_i.data
        if args.adv:
            x_j_adv = gen_adv(model, x_i, criterion)
        
    
        optimizer.zero_grad()
        h_j, z_j = model(x_j)
        loss = criterion(z_i, z_j)
        if args.adv:
            _, z_j_adv = model(x_j_adv, adv=True)
            loss += args.alpha*criterion(z_i, z_j_adv)
        
        loss.backward()

        optimizer.step()
        
        
        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        loss_epoch += loss.item()
        args.global_step += 1
        
        if args.debug:
            break
        
    return loss_epoch



def main():
    args.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'

    root = "../datasets"

    train_sampler = None
    if args.dataset == "CIFAR10":
        root = "../datasets"
        train_dataset = torchvision.datasets.CIFAR10(
            root, download=True, transform=TransformsSimCLR()
        )
        data = 'non_imagenet'
  
    elif args.dataset == "CIFAR100":
        root = "../datasets"
        train_dataset = torchvision.datasets.CIFAR100(
            root, download=True, transform=TransformsSimCLR()
        )
        data = 'non_imagenet'
    elif args.dataset == "tinyImagenet":
        root = '../datasets/tiny_imagenet.pickle'
        train_dataset, _ = load_data(root)
        train_dataset = imagenet(train_dataset, transform=TransformsSimCLR_imagenet(size=224))  
        data = 'imagenet'
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )


    log_dir = "log/" + args.dataset + '_log/'
    
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    suffix = args.dataset + '_{}_batch_{}'.format(args.resnet, args.batch_size)
    if args.adv:
        suffix = suffix + '_alpha_{}_adv_eps_{}'.format(args.alpha, args.eps)
        model, optimizer, scheduler = load_model(args, train_loader, bn_adv_flag = True, bn_adv_momentum = args.bn_adv_momentum, data=data)
    else:
        model, optimizer, scheduler = load_model(args, train_loader, bn_adv_flag = False, bn_adv_momentum = args.bn_adv_momentum, data=data)
 
    suffix = suffix + '_proj_dim_{}'.format(args.projection_dim)
    suffix = suffix + '_bn_adv_momentum_{}_trial_{}'.format(args.bn_adv_momentum, args.trial)
    

    test_log_file = open(log_dir + suffix + '.txt', "w") 
    
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    args.model_dir = args.model_dir + args.dataset + '/'
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
            
    mask = mask_correlated_samples(args)
    criterion = NT_Xent(args.batch_size, args.temperature, mask, args.device)

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(0, args.epochs):
        lr = optimizer.param_groups[0]['lr']
        loss_epoch = train(args, train_loader, model, criterion, optimizer)

        if scheduler:
            scheduler.step()
        print('epoch: {}% \t (loss: {}%)'.format(epoch,  loss_epoch/ len(train_loader)), file = test_log_file)
        test_log_file.flush()
        
        args.current_epoch += 1
        if args.debug:
            break    
    ## end training
    save_model(args.model_dir + suffix, model, optimizer, args.epochs)

if __name__ == "__main__":
    main()
