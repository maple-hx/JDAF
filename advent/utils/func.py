import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('..')
from utils.loss import cross_entropy_2d, cross_entropy_2d_coffecient
from torch.autograd import Variable

EPS = 1e-10

def bce_loss(y_pred, y_label, choice=True):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = Variable(y_truth_tensor).to(y_pred.get_device())
    if choice:
        return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)
    else:
        return nn.BCEWithLogitsLoss(reduction='none')(y_pred, y_truth_tensor)

def ls_loss(y_pred, y_label, choice=True):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    if choice:
        return nn.MSELoss()(y_pred, y_truth_tensor)
    else:
        return nn.MSELoss(reduction='none')(y_pred, y_truth_tensor)

def loss_calc(pred, label, device, choice = False):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    #label = label.long().to(device)
    if choice:
        label = label.long().cuda()
    else:
        label = label.long().cuda(device)
    return cross_entropy_2d(pred, label)

def loss_calc_coffecient(pred, label, entropy_coffecient, device, choice = False):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    if choice:
        label = label.long().cuda()
        entropy_coffecient = entropy_coffecient.cuda()
        
    else:
        label = label.long().cuda(device)
        entropy_coffecient = entropy_coffecient.cuda(device)
    return cross_entropy_2d_coffecient(pred, label, entropy_coffecient)



def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def _adjust_learning_rate(optimizer, i_iter, cfg, learning_rate):
    lr = lr_poly(learning_rate, i_iter, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def _adjust_learning_rate_memory(optimizer, i_iter, cfg, learning_rate):
    lr = lr_poly(learning_rate, i_iter, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.POWER)
    optimizer.param_groups[0]['lr'] = lr * 10
    optimizer.param_groups[1]['lr'] = lr
    optimizer.param_groups[2]['lr'] = lr * 10
    return lr 

def adjust_learning_rate_memory(optimizer, i_iter, cfg):
    """ adject learning rate for main segnet
    """
    lr = _adjust_learning_rate_memory(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE)
    return lr

def adjust_learning_rate(optimizer, i_iter, cfg):
    """ adject learning rate for main segnet
    """
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE)


def adjust_learning_rate_discriminator(optimizer, i_iter, cfg):
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE_D)


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def dice_coefficient(hist):
    """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.
    Args:
        hist: confusion matrix.
    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    A_inter_B = np.diag(hist)
    A = hist.sum(1)
    B = hist.sum(0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    #avg_dice = np.nanmean(dice)
    return dice

def patch_sml(map, random_numlists):
    #random_numlists = torch.rand([2,2])
    #random_numlists[:, 0] = 0.2 + 0.2 * random_numlists[:, 0]
    #random_numlists[:, 1] = 0.6 + 0.2 * random_numlists[:, 1]

    map_H = map.size()[2]
    map_W = map.size()[3]
    numlists = torch.zeros([2,2])
    numlists[0, :] = torch.ceil(random_numlists[0, :] * map_H)
    numlists[1, :] = torch.ceil(random_numlists[1, :] * map_W)
    numlists = numlists.long()
    random_patch = []
    random_numlistsH = torch.cat((torch.cat((torch.tensor([0]), numlists[0])), torch.tensor([map_H])))
    random_numlistsW = torch.cat((torch.cat((torch.tensor([0]), numlists[1])), torch.tensor([map_W])))
    for i in range(1, len(random_numlistsH[1:]) + 1):
        for j in range(1, len(random_numlistsW[1:]) + 1):
            random_patch.append(map[:, :, random_numlistsH[i-1]:random_numlistsH[i], random_numlistsW[j-1]:random_numlistsW[j]])
    
    patch_numel = torch.tensor([x[0,0,:,:].numel() for x in random_patch])
    patch_idex = patch_numel.sort()[1]
    interp = nn.Upsample(size=(map_H, map_W), mode='bilinear', align_corners=True)


    patch_small = [interp(random_patch[patch_idex[0]]),
                   interp(random_patch[patch_idex[1]]),
                   interp(random_patch[patch_idex[2]])]
    patch_medium = [interp(random_patch[patch_idex[3]]),
                    interp(random_patch[patch_idex[4]]),
                    interp(random_patch[patch_idex[5]])]
    patch_large = [interp(random_patch[patch_idex[6]]),
                   interp(random_patch[patch_idex[7]]),
                   interp(random_patch[patch_idex[8]])]

    return patch_small, patch_medium, patch_large