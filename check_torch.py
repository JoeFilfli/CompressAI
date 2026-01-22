import torch

ckpt = torch.load("checkpoint.pth.tar", map_location="cpu")
sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

for k in list(sd.keys())[:30]:
        print(k)

