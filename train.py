import random
from tensorboardX import SummaryWriter
from datetime import datetime
from torch.autograd import Variable
import numpy as np

coding = 'utf-8'
import os
from MFINet import Mnet
from tqdm import tqdm
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.dataset import Data
from lib.data_prefetcher import DataPrefetcher
from torch.nn import functional as F
import pytorch_iou
import pytorch_ssim

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')
bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target):
    pred_S=F.sigmoid(pred)
    bce_out = bce_loss(pred_S,target)
    ssim_out = 1 - ssim_loss(pred_S,target)
    iou_out = iou_loss(pred_S,target)

    loss = bce_out +ssim_out+iou_out
    return loss

def muti_bce_loss_fusion( d1, d2, d3, d4, d5,  labels_v):

	loss1 = bce_ssim_loss(d1,labels_v)
	loss2 = bce_ssim_loss(d2,labels_v)
	loss3 = bce_ssim_loss(d3,labels_v)
	loss4 = bce_ssim_loss(d4,labels_v)
	loss5 = bce_ssim_loss(d5,labels_v)

	loss = loss1+loss2+loss3+loss4+loss5 
	return  loss



if __name__ == '__main__':
    random.seed(825)
    np.random.seed(825)
    torch.manual_seed(825)
    torch.cuda.manual_seed(825)
    torch.cuda.manual_seed_all(825)

    img_root = '/home/cuifengyu/data/VDT-2048_dataset/Train/'
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S-tain/}".format(datetime.now())
    print(TIMESTAMP)
    save_path = './model/' + TIMESTAMP

    lr = 0.00009
    batch_size =4
    epoch = 400
    num_params = 0
    data = Data(img_root)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    net = Mnet().to(device)
    net.load_pretrained_model()
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
     
    iter_num = len(loader)
    net.train()

    train_log_dir = '/home/cuifengyu/Instance/HWSI_finalcode/logs/' + TIMESTAMP
    writer = SummaryWriter(train_log_dir)
    num: int = 0
    min_loss = 5
    for epochi in range(1, epoch + 1):
        r_sal_loss = 0
        epoch_loss = 0
        net.zero_grad()
        i = 0
        for i, sample in tqdm(enumerate(loader), total=len(loader)):
            i = i + 1
            rgb = sample[0].type(torch.FloatTensor).to(device)
            t = sample[1].type(torch.FloatTensor).to(device)
            d = sample[2].type(torch.FloatTensor).to(device)
            label = sample[3].type(torch.FloatTensor).to(device)
            boundary = sample[4].type(torch.FloatTensor).to(device)

            rgb,t,d, label = Variable(rgb, requires_grad=False), Variable(t, requires_grad=False), Variable(d, requires_grad=False), Variable(label,requires_grad=False)
            out1, out2, out3, out4, out5 = net(rgb, t, d)
            sal_loss = muti_bce_loss_fusion(out1, out2, out3, out4, out5, label)

            r_sal_loss += sal_loss.data

            sal_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += sal_loss.data
            if i % 100 == 0:
                print('epoch: [%2d/%2d] ||  loss : %5.4f, lr: %7.6f' % (
                    epochi, epoch, r_sal_loss / 100, lr,))
                r_sal_loss = 0
        print('epoch-%2d_ave_loss: %7.6f' % (epochi, (epoch_loss / i)))
        writer.add_scalar('loss/ep', (epoch_loss / i), epochi)
        if (epochi >= 200) and (epoch_loss/i<min_loss):
            min_loss = epoch_loss / i
            if not os.path.exists(save_path): os.mkdir(save_path)
            torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))
           
    torch.save(net.state_dict(), '%s/final.pth' % save_path)
