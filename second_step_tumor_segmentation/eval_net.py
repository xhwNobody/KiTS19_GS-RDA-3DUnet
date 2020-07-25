from torch.utils.data import DataLoader
from dice_loss import *
from load_data import *

def eval_net(net, valImgPath, valMskPath, valEdgPath, batch_size, gpu=True):
    val_img_list = os.listdir(valImgPath)
    net.eval()
    tot_coef = 0
    tot_loss = 0
    kits = Kits19Dataset(imgPath=valImgPath, mskPath=valMskPath, edgPath=valEdgPath)
    test_loader = DataLoader(kits, batch_size, shuffle=False, num_workers=4)
    for i, data in enumerate(test_loader):
        imgs = data[0]
        true_masks = data[1]
        true_edges = data[2]
        if gpu:
            imgs = imgs.cuda()
            true_masks = true_masks.cuda()
            true_edges = true_edges.cuda()
        mask_pred, edge_pred = net(imgs)
        tot_coef += Multi_class_dice_coef(mask_pred, true_masks).item()
        loss, boundary_loss = Joint_seg_edg_loss(mask_pred, true_masks, edge_pred, true_edges)  ##########
        #tot_loss += Joint_seg_edg_loss(mask_pred, true_masks, edge_pred, true_edges).item()#######
        tot_loss += loss
    return tot_coef / (len(val_img_list)//batch_size), tot_loss/ (len(val_img_list)//batch_size)