import os
from gsplat import PngCompression
import torch
import math
import torch.nn.functional as F


def compress(splats, output_dir, world_rank, device):
    compress_dir = f"{output_dir}/compression/rank{world_rank}"
    os.makedirs(compress_dir, exist_ok=True)
    compression_method = PngCompression()
    compression_method.compress(compress_dir,splats)
    splats_c = compression_method.decompress(compress_dir)
    for k in splats_c.keys():
        splats[k].data = splats_c[k].to(device)
    print(f"Compressed scene saved to {compress_dir}")

def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
    for ckpt_path in args.ckpt:
        ckpt = torch.load(ckpt_path, map_location=device)["splats"]
        means.append(ckpt["means"])
        quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
        scales.append(torch.exp(ckpt["scales"]))
        opacities.append(torch.sigmoid(ckpt["opacities"]))
        sh0.append(ckpt["sh0"])
        shN.append(ckpt["shN"])
    means = torch.cat(means, dim=0)
    quats = torch.cat(quats, dim=0)
    scales = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)
    sh0 = torch.cat(sh0, dim=0)
    shN = torch.cat(shN, dim=0)
    colors = torch.cat([sh0, shN], dim=-2)
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    print("Number of Gaussians:", len(means))
    splats = {
        "means": means, "scales": scales, "quats": quats, "opacities": opacities,
        "sh0": sh0, "shN": shN
    }