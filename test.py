import os
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
from lib.dataset import Data
import torch.nn.functional as F
import torch
import cv2
import time
from MFINet import Mnet
from torchsummary import summary
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')

if __name__ == '__main__':

    TIMESTAMP = "{0:%Y-%m-%d-10-25-904-ssim-340/}".format(datetime.now())
    model_path ="Instance/MFINet/model/MFINet_fIinal.pth"
    data = Data(root='/home/cuifengyu/data/VDT-2048_dataset/Test/', mode='test')
    loader = DataLoader(data, batch_size=1, shuffle=True)
    net = Mnet().to(device)
    out_path = 'Instance/HWSI_finalcode/output/' + TIMESTAMP
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    net.load_state_dict(torch.load(model_path,map_location=torch.device('cuda:0')),strict=True)

    # 参数量和运算量
    # summary(net,input_size=[(3,352,352),(3,352,352),(3,352,352)])
    if False:
        model = net.to(device)
        input1=torch.randn(1,3,352,352).to(device)
        from thop import profile,clever_format
        flops,params=profile(model,inputs=(input1,input1,input1))
        flops,params=clever_format([flops,params],"%.3f")
        print(flops,params)
    
    time_s = time.time()
    img_num = len(loader)
    net.eval()
    num: int = 0
    with torch.no_grad():
        for rgb, t, d, mask_, _, (H, W), name, mask in loader:
            print(name[0])
            score, _, _, _, _ = net(rgb.to(device).float(), t.to(device).float(), d.to(device).float())

            score = F.interpolate(score, size=(H, W), mode='bilinear', align_corners=True)
            pred = np.squeeze(torch.sigmoid(score).cpu().data.numpy())
            pred = (pred - pred.min()) / (pred.max() - pred.min())
            cv2.imwrite(os.path.join(out_path, name[0][:-4] + '.png'), 255 * pred)
    time_e = time.time()
    print('speed: %f FPS' % (img_num / (time_e - time_s)))
