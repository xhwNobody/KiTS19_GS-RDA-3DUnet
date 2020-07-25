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

def Multi_class_dice_loss(input, target):

    return 1. - Multi_class_dice_coef(input, target)












