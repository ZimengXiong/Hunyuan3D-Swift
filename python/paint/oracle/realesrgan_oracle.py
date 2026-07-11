import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
class RDB(nn.Module):
    def __init__(s,nf=64,gc=32):
        super().__init__()
        s.conv1=nn.Conv2d(nf,gc,3,1,1); s.conv2=nn.Conv2d(nf+gc,gc,3,1,1); s.conv3=nn.Conv2d(nf+2*gc,gc,3,1,1)
        s.conv4=nn.Conv2d(nf+3*gc,gc,3,1,1); s.conv5=nn.Conv2d(nf+4*gc,nf,3,1,1); s.l=nn.LeakyReLU(0.2,True)
    def forward(s,x):
        x1=s.l(s.conv1(x)); x2=s.l(s.conv2(torch.cat((x,x1),1))); x3=s.l(s.conv3(torch.cat((x,x1,x2),1)))
        x4=s.l(s.conv4(torch.cat((x,x1,x2,x3),1))); x5=s.conv5(torch.cat((x,x1,x2,x3,x4),1)); return x5*0.2+x
class RRDB(nn.Module):
    def __init__(s,nf=64,gc=32):
        super().__init__(); s.rdb1=RDB(nf,gc); s.rdb2=RDB(nf,gc); s.rdb3=RDB(nf,gc)
    def forward(s,x): return s.rdb3(s.rdb2(s.rdb1(x)))*0.2+x
class RRDBNet(nn.Module):
    def __init__(s,nb=23,nf=64,gc=32):
        super().__init__()
        s.conv_first=nn.Conv2d(3,nf,3,1,1); s.body=nn.Sequential(*[RRDB(nf,gc) for _ in range(nb)])
        s.conv_body=nn.Conv2d(nf,nf,3,1,1); s.conv_up1=nn.Conv2d(nf,nf,3,1,1); s.conv_up2=nn.Conv2d(nf,nf,3,1,1)
        s.conv_hr=nn.Conv2d(nf,nf,3,1,1); s.conv_last=nn.Conv2d(nf,3,3,1,1); s.l=nn.LeakyReLU(0.2,True)
    def forward(s,x):
        feat=s.conv_first(x); feat=feat+s.conv_body(s.body(feat))
        feat=s.l(s.conv_up1(F.interpolate(feat,scale_factor=2,mode='nearest')))
        feat=s.l(s.conv_up2(F.interpolate(feat,scale_factor=2,mode='nearest')))
        return s.conv_last(s.l(s.conv_hr(feat)))
d=sys.argv[1]; os.makedirs(d,exist_ok=True)
m=RRDBNet(); sd=torch.load('weights/realesrgan/RealESRGAN_x4plus.pth',map_location='cpu',weights_only=True)
m.load_state_dict(sd.get('params_ema',sd.get('params',sd)),strict=True); m.eval().float()
torch.manual_seed(0); x=torch.rand(1,3,64,64)
with torch.no_grad(): y=m(x)
np.save(d+'/in.npy', x.numpy().transpose(0,2,3,1)); np.save(d+'/out.npy', y.numpy().transpose(0,2,3,1))
print('OK in',tuple(x.shape),'-> out',tuple(y.shape))
