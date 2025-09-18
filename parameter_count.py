# save as count_params.py and run: python count_params.py
import torch
from torch import nn

# make sure your repo is on PYTHONPATH or run this from the repo root
from servir.methods.ldm.generative.networks.nets.diffusion_model_unet import DiffusionModelUNet

def count_params(module: nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def human(n):
    for unit in ['','K','M','B']:
        if abs(n) < 1000:
            return f"{n:.0f}{unit}"
        n /= 1000.0
    return f"{n:.1f}B"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=32,
    out_channels=32,
    num_res_blocks=6,                     # 6 per level (down); decoder uses +1 per level internally
    num_channels=(128, 256, 512, 512),    # four scales
    attention_levels=(False, True, True, True),
    num_head_channels=(0, 256, 512, 512),
    with_conditioning=True,
    cross_attention_dim=64,
).to(device)

total, trainable = count_params(model)

# Log output to file and print to console
output_lines = [
    f"Total params:     {total:,}  ({human(total)})",
    f"Trainable params: {trainable:,}  ({human(trainable)})",
    "\n# Per-submodule breakdown:"
]

# Optional: per-submodule breakdown
def report(mod: nn.Module, prefix="", output_list=None):
    if output_list is None:
        output_list = []
    subtotal = sum(p.numel() for p in mod.parameters())
    line = f"{prefix}{mod.__class__.__name__}: {subtotal:,}"
    output_list.append(line)
    for name, child in mod.named_children():
        report(child, prefix + "  ", output_list)
    return output_list

# Get detailed breakdown
breakdown_lines = report(model)
output_lines.extend(breakdown_lines)

# Write to file
with open('parameter_count_output.txt', 'w') as f:
    for line in output_lines:
        f.write(line + '\n')
        print(line)

print(f"\nOutput saved to parameter_count_output.txt")
