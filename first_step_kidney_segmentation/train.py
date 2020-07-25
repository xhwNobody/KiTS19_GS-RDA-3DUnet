import sys
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from torch import optim
from optparse import OptionParser
from build_RDA_3DUnet import UNet
from eval_net import *

def train_net(net,
              epochs=5,
              subnums=5,
              batch_size=2,
              lr=0.1,
              save_cp=True,
              gpu=True):
    train_img_dir = './train_Image/'
    train_msk_dir = './train_Mask/'
    val_img_dir = './val_Image/'
    val_msk_dir = './val_Mask/'
    dir_checkpoint = './checkpoints/'

    train_img_list = os.listdir(train_img_dir)
    val_img_list = os.listdir(val_img_dir)

    print('''
Starting training:
    Training size: {}
    Validation size: {}
    Epochs: {}
    Subnums: {}
    Batch size: {}
    Learning rate: {}
    Checkpoints: {}
    CUDA: {}
    '''.format(len(train_img_list), len(val_img_list), epochs, subnums, batch_size, lr, str(save_cp), str(gpu)))

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20], gamma=0.1)

    info_txt = open('train-val-info.txt', 'w')
    epoch_loss_list = []
    val_loss_list = []
    max_val_dice_coef = 0
    iter_num = 0
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch, epochs))
        lr_scheduler.step()
        print('===>LR:',optimizer.param_groups[0]['lr'])
        net.train()
        epoch_loss = 0
        sub_iter_num = 0
        kits = Kits19Dataset(imgPath=train_img_dir, mskPath=train_msk_dir)
        train_loader = DataLoader(kits, batch_size, shuffle=True, num_workers=4)

        for i, data in enumerate(train_loader):
            info_list = []
            imgs = data[0]
            true_masks = data[1]
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()
            masks_pred = net(imgs)
            loss = Multi_class_dice_loss(masks_pred, true_masks)
            print('iter:' + str(sub_iter_num) + ' ' + str('%4f' % loss.item()))
            sub_iter_num += 1
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % subnums == 0 and i != 0:
                print('Epoch {} SubEpoch {} finished ! Loss: {}'.format(epoch, iter_num, epoch_loss / subnums))
                epoch_loss_list.append(epoch_loss / subnums)
                info_list.append(epoch_loss / subnums)

                with torch.no_grad():
                    val_dice, val_loss = eval_net(net, val_img_dir, val_msk_dir, batch_size,  gpu)
                    val_loss_list.append(val_loss)
                    info_list.append(val_loss)
                    info_list.append(val_dice)
                    print('Validation Dice Coeff: {}'.format(val_dice))

                if save_cp and val_dice>max_val_dice_coef:
                    torch.save(net.state_dict(), dir_checkpoint + 'CP{}_val_{}.pth'.format(iter_num, '%4f' % val_dice))
                    print('Checkpoint {} saved !'.format(iter_num))
                    max_val_dice_coef = val_dice
                iter_num += 1
                epoch_loss = 0
                sub_iter_num = 0
                info_txt.writelines('train-loss:' + str(info_list[0]) + ' ' + 'val-loss:' + str(info_list[1]) + ' ' + 'val-dice-acc:' + str(info_list[2]) + '\n')

    #绘制损失函数曲线
    plt.title('train-val loss')
    x = np.array(list(range(len(epoch_loss_list))))
    y_1 = np.array(epoch_loss_list)
    y_2 = np.array(val_loss_list)
    x_new = np.linspace(0,len(epoch_loss_list)-1,epochs*(len(train_img_list)//batch_size//subnums))
    y1_smooth = spline(x, y_1, x_new)
    y2_smooth = spline(x, y_2, x_new)
    plt.plot(x_new, y1_smooth, color = 'green', label = 'train loss')
    plt.plot(x_new, y2_smooth, color = 'red', label = 'val loss')
    plt.legend()
    plt.savefig('record.png')

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=25, type='int',
                      help='number of epochs')
    parser.add_option('-n', '--sub-nums', dest='subnums', default=300, type='int',
                      help='number part of epoch')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=2,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.0001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=1, n_classes=3)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  subnums=args.subnums,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
