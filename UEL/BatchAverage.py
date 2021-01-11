import torch
from torch.autograd import Function
from torch import nn
import math
import numpy as np
import pdb

def gen_mask(k, feat_dim):
    mask = None
    for i in range(k):
        tmp_mask = torch.triu(torch.randint(0, 2, (feat_dim, feat_dim)), 1)
        tmp_mask = tmp_mask + torch.triu(1-tmp_mask,1).t()
        tmp_mask = tmp_mask.view(tmp_mask.shape[0], tmp_mask.shape[1],1)
        mask = tmp_mask if mask is None else torch.cat([mask,tmp_mask],2)
    return mask

def entropy(prob):
    # assume m x m x k input
    return -torch.sum(prob*torch.log(prob),1)

class BatchCriterion(nn.Module):
    ''' Compute the loss within each batch  
    '''
    def __init__(self, negM, T, batchSize):
        super(BatchCriterion, self).__init__()
        self.negM = negM
        self.T = T
        self.diag_mat = 1 - torch.eye(batchSize*2).cuda()
       
    def forward(self, x, targets):
        batchSize = x.size(0)
        
        #get positive innerproduct
        reordered_x = torch.cat((x.narrow(0,batchSize//2,batchSize//2),\
                x.narrow(0,0,batchSize//2)), 0)
        
        
        #reordered_x = reordered_x.data
        pos = (x*reordered_x.data).sum(1).div_(self.T).exp_()

        #get all innerproduct, remove diag
        all_prob = torch.mm(x,x.t().data).div_(self.T).exp_()*self.diag_mat
        if self.negM==1:
            all_div = all_prob.sum(1)
        else:
            #remove pos for neg
            all_div = (all_prob.sum(1) - pos)*self.negM + pos

        lnPmt = torch.div(pos, all_div)

        # negative probability
        Pon_div = all_div.repeat(batchSize,1)
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)
   
        
        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()

        
        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        # negative multiply m
        lnPonsum = lnPonsum * self.negM
        loss = - (lnPmtsum + lnPonsum)/batchSize

        return loss
