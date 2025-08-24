# inspect
from safetensors.torch import load_file
sd = load_file("outputs/train/vla/checkpoints/000100/pretrained_model/model.safetensors", device="cpu")
print("keys:", len(sd), list(sd.keys())[:60])
for k,v in sd.items():
    print(k, tuple(v.shape))
    break

# convert to .pt
import torch
new_sd = {}
for k,v in sd.items():
    nk = k
    if nk.startswith("network."):
        nk = nk[len("network."):]
    if nk.startswith("module."):
        nk = nk[len("module."):]
    new_sd[nk] = v
torch.save({"model_state_dict": new_sd}, "outputs/train/vla/checkpoints/000100/pretrained_model/model_converted.pt")
