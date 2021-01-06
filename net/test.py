import os,argparse
import numpy as np
from PIL import Image
from net.models import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from net.models.FFA import FFA
import glob

abs=os.getcwd()+'/'
def tensorShow(tensors,titles=['haze']):
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(221+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

parser=argparse.ArgumentParser()
parser.add_argument('--task',type=str,default='its',help='its or ots')
parser.add_argument('--test_imgs',type=str,default='test_imgs',help='Test imgs folder')
opt=parser.parse_args()
dataset=opt.task
gps=3
blocks=19
#img_dir=abs+opt.test_imgs+'/'
#img_dir = "E:/Hazy Dataset Benchmark/O-HAZE/hazy/"
img_dir = "E:/Hazy Dataset Benchmark/RESIDE-Unannotated/"
hazy_list = glob.glob(img_dir + "*.jpeg")

output_dir=abs+f'pred_FFA_{dataset}/'
print("pred_dir:",output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_dir=abs+f'trained_models/{dataset}_train_ffa_{gps}_{blocks}.pk'
device='cuda' if torch.cuda.is_available() else 'cpu'
ckp=torch.load(model_dir,map_location=device)
net=FFA(gps=gps,blocks=blocks)
net=nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()
for im in hazy_list:
    img_name = im.split("\\")[1]
    print(f'\r {im}',end='',flush=True)
    haze = Image.open(im)
    #resize images
    width, height = haze.size
    #haze = haze.resize((int(width / 2), int(height / 2)), Image.BILINEAR)
    haze = haze.resize((512, 512), Image.BILINEAR)

    haze1= tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152]) #normalize with the dataset mean?
    ])(haze)[None,::]
    haze_no=tfs.ToTensor()(haze)[None,::]
    with torch.no_grad():
        pred = net(haze1)
    ts=torch.squeeze(pred.clamp(0,1).cpu())
    tensorShow([haze_no,pred.clamp(0,1).cpu()],['haze','pred'])

    vutils.save_image(ts,output_dir+img_name+'_FFA.jpg')
