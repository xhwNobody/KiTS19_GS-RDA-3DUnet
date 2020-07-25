import torch
import torch.nn as nn
import torch.nn.functional as F

class dice_coef(nn.Module):
    def __init__(self):
            super(dice_coef, self).__init__()

    def forward(self, input, target):
        smooth = 0.000001
        intersection = torch.sum(input*target, dim=(2, 3, 4))
        union = torch.sum(input, dim=(2, 3, 4)) + torch.sum(target, dim=(2, 3, 4))
        sample_dicess = (2. * intersection + smooth) / (union + smooth)
        dices = torch.mean(sample_dicess, 0)
        return torch.mean(dices)

def Multi_class_dice_coef(input, target):

    return dice_coef().forward(input, target)

# def Multi_class_dice_loss(input, target):
#     return 1. - Multi_class_dice_coef(input, target)

# def FocalLoss(input, target, alpha=0.25, gamma=2.0):
#     input = input.view(input.size(0), input.size(1), -1)
#     input = input.transpose(1, 2)
#     input = input.contiguous().view(-1, input.size(2))
#     target = target.contiguous().view(target.size(0), target.size(1), -1)
#     target = target.transpose(1, 2)
#     target = target.contiguous().view(-1, target.size(2))
#     logpt = -F.binary_cross_entropy(input, target)
#     pt = torch.exp(logpt)
#     loss = -((1-pt) ** gamma) * logpt
#     return loss.mean()


def Joint_seg_edg_loss(input, target, edg_input, edg_target):

    return 1. - Multi_class_dice_coef(input, target) + F.binary_cross_entropy(edg_input, edg_target), F.binary_cross_entropy(edg_input, edg_target)


# def Joint_seg_edg_loss(input, target, edg_input, edg_target):
#
#     return FocalLoss(edg_input, edg_target) +(- torch.log(1. - Multi_class_dice_coef(input, target))), FocalLoss(edg_input, edg_target)
#     #return FocalLoss(edg_input, edg_target), -torch.log(1. - Multi_class_dice_coef(input, target))










