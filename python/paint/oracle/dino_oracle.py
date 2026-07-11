import os, sys, numpy as np, torch
from transformers import AutoModel
d=sys.argv[1]; os.makedirs(d, exist_ok=True)
m=AutoModel.from_pretrained('weights/dinov2-giant').eval().float()
torch.manual_seed(0)
pix=torch.randn(1,3,518,518)
with torch.no_grad():
    out=m(pix)[0]  # last_hidden_state [1,1370,1536]
np.save(f'{d}/pix.npy', pix.numpy())
np.save(f'{d}/out.npy', out.numpy())
print('OK dino out', tuple(out.shape))
