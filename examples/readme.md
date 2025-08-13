# Usage

## training 

- gsplat

```
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --data_dir data/360_v2/garden/ --data_factor 4 \
    --result_dir ./results/garden
```

- 3dgs

please refer to [3d gaussian splatting](https://github.com/graphdeco-inria/gaussian-splatting) for more information.

## rendering

- gsplat backend

```
# View it in a viewer with gsplat rasterization
python simple_viewer.py --scene_grid 5 --ckpt results/garden/ckpts/ckpt_6999.pt --camera_path results/garden/colmap/colmap_camera.npy --backend gsplat --train_mode gsplat
```

- 3dgs backend

```
# View it in a viewer with 3dgs
python examples/simple_viewer.py --ckpt results_3dgs/train/point_cloud/iteration_30000/point_cloud.ply --camera_path results_3dgs/train/cameras.json --backend 3dgs --train_mode 3dgs
```

## rendering with diffusion shader

