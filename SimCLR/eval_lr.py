import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import os
from model import load_model, save_model

from modules import LogisticRegression
from load_imagenet import imagenet, load_data

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


def train(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        
        x = x.to(args.device)
        y = y.to(args.device)

        # get encoding
        with torch.no_grad():
            h, z = simclr_model(x)
            # h = 512
            # z = 64

        output = model(h)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        if step % 100 == 0:
            print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}")
        
        if args.debug:
            break
        
    return loss_epoch, accuracy_epoch

def test(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        # get encoding
        with torch.no_grad():
            h, z = simclr_model(x)
            # h = 512
            # z = 64

        output = model(h)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()


    return loss_epoch, accuracy_epoch

def main():
    args.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'

    root = "../datasets"

    if args.dataset == 'tinyImagenet':
        transform = transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
       ])
        data = 'imagenet'
    else:
        transform = transforms.Compose([
        torchvision.transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        data = 'non_imagenet'

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
    elif args.dataset == "tinyImagenet":
        root = '../datasets/tiny_imagenet.pickle'
        train_dataset, test_dataset = load_data(root)
        train_dataset = imagenet(train_dataset, transform=transform)
        test_dataset = imagenet(test_dataset, transform=transform)
    else:
        raise NotImplementedError

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

    log_dir = "log_eval/" + args.dataset + '_LR_log/'
    
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    suffix = args.dataset + '_{}_batch_{}'.format(args.resnet, args.batch_size)
    if args.adv:
        suffix = suffix + '_alpha_{}_adv_eps_{}'.format(args.alpha, args.eps)
 
    suffix = suffix + '_proj_dim_{}'.format(args.projection_dim)
    suffix = suffix + '_bn_adv_momentum_{}_trial_{}'.format(args.bn_adv_momentum, args.trial)
    
    args.model_dir = args.model_dir + args.dataset + '/'
    print("Loading {}".format(args.model_dir + suffix + '_epoch_100.pt'))
    if args.adv:
        simclr_model, _, _ = load_model(args, train_loader, reload_model=True , load_path = args.model_dir + suffix + '_epoch_100.pt', bn_adv_flag = True, bn_adv_momentum = args.bn_adv_momentum, data=data)
    else:
        simclr_model, _, _ = load_model(args, train_loader, reload_model=True , load_path = args.model_dir + suffix + '_epoch_100.pt', bn_adv_flag = False, bn_adv_momentum = args.bn_adv_momentum, data=data)
 

    test_log_file = open(log_dir + suffix + '.txt', "w") 
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()


    ## Logistic Regression
    if args.dataset == "CIFAR100":
        n_classes = 100 # stl-10
    elif args.dataset == 'tinyImagenet':
        n_classes = 200
    else:
        n_classes = 10

    model = LogisticRegression(simclr_model.n_features, n_classes)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()


    best_acc = 0
    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = train(args, train_loader, simclr_model, model, criterion, optimizer)
        print("Train Epoch [{}]\t Loss: {}\t Accuracy: {}".format(epoch, loss_epoch / len(train_loader), accuracy_epoch / len(train_loader)), file = test_log_file)
        test_log_file.flush()
        

        # final testing
        test_loss_epoch, test_accuracy_epoch = test(args, test_loader, simclr_model, model, criterion, optimizer)
        test_current_acc = test_accuracy_epoch / len(test_loader)
        if test_current_acc > best_acc:
            best_acc = test_current_acc
        print("Test Epoch [{}]\t Loss: {}\t Accuracy: {}\t Best Accuracy: {}".format(epoch, test_loss_epoch / len(test_loader), test_current_acc, best_acc), file = test_log_file)
        test_log_file.flush()
        
        if args.debug:
            break 
    print("Final \t Best Accuracy: {}".format(epoch, best_acc), file = test_log_file)
    test_log_file.flush()
    if not os.path.isdir("checkpoint/" + args.dataset + '_eval/'):
        os.makedirs("checkpoint/" + args.dataset + '_eval/')
    save_model("checkpoint/" + args.dataset + '_eval/' + suffix, model, optimizer, 0)
    
    
    
if __name__ == "__main__":
    main()
