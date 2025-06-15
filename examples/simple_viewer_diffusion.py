import argparse
import math
import os
import time
import json
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser
from pathlib import Path
from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization, rasterization_inria_wrapper
from plyfile import PlyData
import PIL.Image
from wrapper import StreamDiffusionWrapper

from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat_viewer_diffusion import GsplatViewer, GsplatRenderTabState

def load_ply_to_gsplat_vars(path, max_sh_degree=3, device="cuda"):
    plydata = PlyData.read(path)

    # means
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    means = torch.tensor(xyz, dtype=torch.float, device=device)

    # opacities
    opacities = np.asarray(plydata.elements[0]["opacity"])
    opacities = torch.tensor(opacities, dtype=torch.float, device=device).squeeze()
    opacities = torch.sigmoid(opacities)

    # sh0
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    sh0 = torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous()

    # shN
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    shN = torch.tensor(features_extra, dtype=torch.float, device=device).transpose(1, 2).contiguous()

    # scales
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    scales = torch.tensor(scales, dtype=torch.float, device=device)
    scales = torch.exp(scales)

    # quats
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    quats = torch.tensor(rots, dtype=torch.float, device=device)
    quats = F.normalize(quats, p=2, dim=-1)

    return means, quats, scales, opacities, sh0, shN


def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    if args.ckpt is None:
        (
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width,
            height,
        ) = load_test_data(device=device, scene_grid=args.scene_grid)

        assert world_size <= 2
        means = means[world_rank::world_size].contiguous()
        means.requires_grad = True
        quats = quats[world_rank::world_size].contiguous()
        quats.requires_grad = True
        scales = scales[world_rank::world_size].contiguous()
        scales.requires_grad = True
        opacities = opacities[world_rank::world_size].contiguous()
        opacities.requires_grad = True
        colors = colors[world_rank::world_size].contiguous()
        colors.requires_grad = True

        viewmats = viewmats[world_rank::world_size][:1].contiguous()
        Ks = Ks[world_rank::world_size][:1].contiguous()

        sh_degree = None
        C = len(viewmats)
        N = len(means)
        print("rank", world_rank, "Number of Gaussians:", N, "Number of Cameras:", C)

        # batched render
        for _ in tqdm.trange(1):
            render_colors, render_alphas, meta = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, S, 3]
                viewmats,  # [C, 4, 4]
                Ks,  # [C, 3, 3]
                width,
                height,
                render_mode="RGB+D",
                packed=False,
                distributed=world_size > 1,
            )
        C = render_colors.shape[0]
        assert render_colors.shape == (C, height, width, 4)
        assert render_alphas.shape == (C, height, width, 1)
        render_colors.sum().backward()

        render_rgbs = render_colors[..., 0:3]
        render_depths = render_colors[..., 3:4]
        render_depths = render_depths / render_depths.max()

        # dump batch images
        os.makedirs(args.output_dir, exist_ok=True)
        canvas = (
            torch.cat(
                [
                    render_rgbs.reshape(C * height, width, 3),
                    render_depths.reshape(C * height, width, 1).expand(-1, -1, 3),
                    render_alphas.reshape(C * height, width, 1).expand(-1, -1, 3),
                ],
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        imageio.imsave(
            f"{args.output_dir}/render_rank{world_rank}.png",
            (canvas * 255).astype(np.uint8),
        )
    else:
        means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []

        file_type = args.ckpt[0].split(".")[-1]

        if file_type == "pt":
            for ckpt_path in args.ckpt:
                ckpt = torch.load(ckpt_path, map_location=device)["splats"]
                means.append(ckpt["means"])
                quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
                scales.append(torch.exp(ckpt["scales"]))
                opacities.append(torch.sigmoid(ckpt["opacities"]))
                sh0.append(ckpt["sh0"])
                shN.append(ckpt["shN"])
        elif file_type == "ply":
            for ckpt_path in args.ckpt:
                _means, _quats, _scales, _opacities, _sh0, _shN = load_ply_to_gsplat_vars(ckpt_path, device=device)
                means.append(_means)
                quats.append(_quats)
                scales.append(_scales)
                opacities.append(_opacities)
                sh0.append(_sh0)
                shN.append(_shN)

        means = torch.cat(means, dim=0)
        quats = torch.cat(quats, dim=0)
        scales = torch.cat(scales, dim=0)
        opacities = torch.cat(opacities, dim=0)
        sh0 = torch.cat(sh0, dim=0)
        shN = torch.cat(shN, dim=0)
        colors = torch.cat([sh0, shN], dim=-2)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

        print("Number of Gaussians:", len(means))
    
    if args.camera_path is not None:
        if args.train_mode == "gsplat":
            colmap_camera_data = np.load(args.camera_path, allow_pickle=True).item()
            Ks = colmap_camera_data["Ks"]
            camtoworlds = colmap_camera_data["camtoworlds"]
            init_camera_extrinsics = CameraState(
                c2w=camtoworlds[0],
                fov=50,
                aspect=None,
            )
        elif args.train_mode == "3dgs":
            colmap_camera_data = json.load(open(args.camera_path))[0]
            rotate = np.array(colmap_camera_data["rotation"])
            trans = np.array(colmap_camera_data["position"])
            c2w = np.eye(4)
            c2w[:3, :3] = rotate
            c2w[:3, 3] = trans
            init_camera_extrinsics = CameraState(
                c2w=c2w,
                fov=50,
                aspect=None,
            )
        else:
            raise ValueError
    else:
        init_camera_extrinsics = None



    # === StreamDiffusionWrapper Initialization ===
    if args.diffusion:
        stream = StreamDiffusionWrapper(
            model_id_or_path="KBlueLeaf/kohaku-v2.1",
            t_index_list=[32, 45],
            mode="img2img",
            width=512,
            height=512,
            frame_buffer_size=1,
            # acceleration="xformers",
            acceleration="tensorrt",
            use_lcm_lora=True,
            use_tiny_vae=True,
            device="cuda",
            dtype=torch.float16,
            seed=2,
        )
        stream.prepare(
            prompt="a low quality scene need to be improved",
            negative_prompt="bad image , bad quality",
            num_inference_steps=50,
            guidance_scale=1.4,
            delta=0.5,
        )
        first_render = None
        width = 512
        height = 512
        aspect = 1.0
        c2w = torch.from_numpy(init_camera_extrinsics.c2w).float().to(device)
        K = torch.from_numpy(init_camera_extrinsics.get_K((width, height))).float().to(device)
        viewmat = c2w.inverse()
        first_render, _, _ = rasterization(
                    means,  # [N, 3]
                    quats,  # [N, 4]
                    scales,  # [N, 3]
                    opacities,  # [N]
                    colors,  # [N, S, 3]
                    viewmat[None],  # [1, 4, 4]
                    K[None],  # [1, 3, 3]
                    width,
                    height,
                    sh_degree=3
                )
        first_render = first_render[0, ..., 0:3].clamp(0, 1) # [H, W, 3]
        first_render = first_render.permute(2, 0, 1) # [3, H, W]
        first_render = first_render.unsqueeze(0) # [1, 3, H, W]
        warmup = 10
        for _ in range(warmup):
            stream.stream(first_render)
    else:
        stream = None



    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        start_time = time.time()
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        
        aspect = camera_state.aspect
        max_img_res = render_tab_state.viewer_res
        max_H = max_img_res
        max_W = int(max_H * aspect)
        if max_W > max_img_res:
            max_W = max_img_res
            max_H = int(max_W / aspect)
        
        img_canvas = np.zeros((max_H, max_W, 3))

        # Restrict the image size to 512x512
        width = 512
        height = 512


        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }
        # print(width, height, camera_state.aspect)
        if args.backend == "gsplat":
            render_colors, render_alphas, info = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, S, 3]
                viewmat[None],  # [1, 4, 4]
                K[None],  # [1, 3, 3]
                width,
                height,
                sh_degree=(
                    min(render_tab_state.max_sh_degree, sh_degree)
                    if sh_degree is not None
                    else None
                ),
                near_plane=render_tab_state.near_plane,
                far_plane=render_tab_state.far_plane,
                radius_clip=render_tab_state.radius_clip,
                eps2d=render_tab_state.eps2d,
                backgrounds=torch.tensor([render_tab_state.backgrounds], device=device)
                / 255.0,
                render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
                rasterize_mode=render_tab_state.rasterize_mode,
                camera_model=render_tab_state.camera_model,
                packed=False,
                with_ut=args.with_ut,
                with_eval3d=args.with_eval3d,
            )
        elif args.backend == "3dgs":
            render_colors, render_alphas, info = rasterization_inria_wrapper(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, S, 3]
                viewmat[None],  # [1, 4, 4]
                K[None],  # [1, 3, 3]
                width,
                height,
                near_plane=render_tab_state.near_plane,
                far_plane=render_tab_state.far_plane,
                backgrounds=torch.tensor([render_tab_state.backgrounds], device=device)
                / 255.0,
                sh_degree=(
                    min(render_tab_state.max_sh_degree, sh_degree)
                    if sh_degree is not None
                    else None
                ),
                rasterize_mode=render_tab_state.rasterize_mode,
            )
        else:
            raise ValueError
        end_time = time.time()
        render_tab_state.total_gs_count = len(means)
        if ("radii" in info) and (info["radii"] is not None):
            render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            render_colors = render_colors.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]

            if stream is not None:
                render_colors = stream.stream(render_colors) # [1, 3, H, W]


            width = min(width, max_W)
            height = min(height, max_H)

            render_colors = F.interpolate(render_colors, size=(height, width), mode="bilinear", align_corners=False)
            render_colors = render_colors.permute(0, 2, 3, 1).squeeze(0) # [H, W, 3]
            # end_time = time.time()

            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha" and render_alphas is not None:
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )



        img_canvas[:height, :width, :] = renders
        # end_time = time.time()
        render_tab_state.fps_render = 1.0 / (end_time - start_time)
        return img_canvas

    server = viser.ViserServer(port=args.port, verbose=False)
    _ = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
        init_camera_extrinsics=init_camera_extrinsics,
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --ckpt results/garden/ckpts/ckpt_6999_rank0.pt \
        --output_dir results/garden/ \
        --port 8082
    
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --output_dir results/garden/ \
        --port 8082
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    ),
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt / .ply file"
    )
    parser.add_argument("--train_mode", type=str, default='gsplat', help="gsplat, 3dgs")
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--with_ut", action="store_true", help="use uncentered transform"
    )
    parser.add_argument("--camera_path", type=str, default=None, help="path to the camera file")
    parser.add_argument("--with_eval3d", action="store_true", help="use eval 3D")

    parser.add_argument("--backend", type=str, default="gsplat", help="gsplat, 3dgs")
    parser.add_argument("--compress", action="store_true", help="compress the scene")
    parser.add_argument("--diffusion", action="store_true", help="use diffusion")
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
