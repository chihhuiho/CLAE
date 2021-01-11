import torch
import torchvision
import torchvision.transforms as transforms
import argparse

import os
#from experiment import ex
from model import load_model, save_model

from modules import LogisticRegression
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch Seen Testing Category Training')
parser.add_argument('--batch_size', default=256, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--logistic_batch_size', default=256, type=int,
                    metavar='B', help='logistic_batch_size batch size')
parser.add_argument('--logistic_epochs', default=1000, type=int, help='logistic_epochs')
parser.add_argument('--workers', default=4, type=int, help='workers')
parser.add_argument('--epochs', default=100, type=int,help='epochs')
parser.add_argument('--resnet', default="resnet18", type=str, help="resnet")
parser.add_argument('--normalize', default=True, action='store_true', help='normalize')
parser.add_argument('--projection_dim', default=64, type=int,help='projection_dim')
parser.add_argument('--optimizer', default="Adam", type=str, help="optimizer")
parser.add_argument('--weight_decay', default=1.0e-6, type=float, help='weight_decay')
parser.add_argument('--temperature', default=0.5, type=float, help='temperature')
parser.add_argument('--model_path', default='checkpoint/', type=str, 
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


def kNN(epoch, net, trainloader, testloader, K, sigma, ndata, low_dim = 128):
    net.eval()
    total = 0
    correct_t = 0
    testsize = testloader.dataset.__len__()

   
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        try:
            trainLabels = torch.LongTensor(trainloader.dataset.targets).cuda()
        except:
            trainLabels = torch.LongTensor(trainloader.dataset.labels).cuda()
    trainFeatures = np.zeros((low_dim, ndata))
    trainFeatures = torch.Tensor(trainFeatures).cuda() 
    C = trainLabels.max() + 1
    C = np.int(C)
    
    with torch.no_grad(): 
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=256, shuffle=False, num_workers=4)
        for batch_idx, (inputs, targets) in tqdm(enumerate(temploader)):
            targets = targets.cuda()
            batchSize = inputs.size(0)
            _, features = net(inputs.cuda())
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.t()
            
            
    trainloader.dataset.transform = transform_bak
    # 
    
       
    top1 = 0.
    top5 = 0.
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            
            targets = targets.cuda()
            batchSize = inputs.size(0)  
            _, features = net(inputs.cuda())
            total += targets.size(0)

            dist = torch.mm(features, trainFeatures)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)

            _, predictions = probs.sort(1, True)
            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            
    print(top1*100./total)

    return top1*100./total


def test(args, trainloader, testloader, net, ndata):
    net.eval()
    print('----------Evaluation---------')
    #acc = kNN(0, net, trainloader, testloader, 200, args.temperature, ndata, low_dim = net.n_features)
    acc = kNN(0, net, trainloader, testloader, 200, args.temperature, ndata, low_dim = args.projection_dim)
        
    return acc

def main():
    args.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'

    root = "../datasets"
    transform = transforms.Compose([
        torchvision.transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if args.dataset == "CIFAR10" :
        train_dataset = torchvision.datasets.CIFAR10(
            root, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root, train=False, download=True, transform=transform
        )
    elif args.dataset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(
            root, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root, train=False, download=True, transform=transform
        )
    else:
        raise NotImplementedError

    ndata = train_dataset.__len__()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    log_dir = "log_eval/" + args.dataset + '_knn_log/'
    
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    
    suffix = args.dataset + '_{}_batch_{}'.format(args.resnet, args.batch_size)
    if args.adv:
        suffix = suffix + '_alpha_{}_adv_eps_{}'.format(args.alpha, args.eps)
 
    suffix = suffix + '_proj_dim_{}'.format(args.projection_dim)
    suffix = suffix + '_bn_adv_momentum_{}_trial_{}'.format(args.bn_adv_momentum, args.trial)
    
    args.model_dir = args.model_dir + args.dataset + '/'
    if args.adv:
        simclr_model, _, _ = load_model(args, train_loader, reload_model=True , load_path = args.model_dir + suffix + '_epoch_100.pt', bn_adv_flag = True, bn_adv_momentum = args.bn_adv_momentum)
    else:
        simclr_model, _, _ = load_model(args, train_loader, reload_model=True , load_path = args.model_dir + suffix + '_epoch_100.pt', bn_adv_flag = False, bn_adv_momentum = args.bn_adv_momentum)
 

    test_log_file = open(log_dir + suffix + '.txt', "w") 
    print("Loading {}".format(args.model_dir + suffix + '_epoch_100.pt'))
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()


    ## Logistic Regression
    if args.dataset == "CIFAR100":
        n_classes = 100 # stl-10
    else:
        n_classes = 10
    
    acc = test(args, train_loader, test_loader, simclr_model, ndata)

    print("Final \t Best Accuracy: {}".format(acc), file = test_log_file)
    test_log_file.flush()
    
    
    
if __name__ == "__main__":
    main()
