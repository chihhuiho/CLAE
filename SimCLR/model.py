import os
import torch
from modules import SimCLR_BN


def load_model(args, loader, reload_model=False, load_path = None, bn_adv_flag=False, bn_adv_momentum = 0.01, data='non_imagenet'):

    model = SimCLR_BN(args, bn_adv_flag=bn_adv_flag, bn_adv_momentum = bn_adv_momentum, data = data)

    if reload_model:
        if os.path.isfile(load_path):
            model_fp = os.path.join(load_path)
        else:
            print("No file to load")
            return
        model.load_state_dict(torch.load(model_fp, map_location=lambda storage, loc: storage))
        
    model = model.to(args.device)

    scheduler = None
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # TODO: LARS

    return model, optimizer, scheduler


def save_model(model_dir, model, optimizer, epoch):
   

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), model_dir + '_epoch_{}.pt'.format(epoch))
    else:
        torch.save(model.state_dict(), model_dir + '_epoch_{}.pt'.format(epoch))
