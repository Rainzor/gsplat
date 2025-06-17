import io
import os
import sys
from typing import List, Literal, Optional, Dict

import fire
import PIL.Image
import requests
import torch
from tqdm import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from wrapper import StreamDiffusionWrapper
from streamdiffusion.image_utils import postprocess_image

def download_image(url: str):
    response = requests.get(url)
    image = PIL.Image.open(io.BytesIO(response.content))
    return image
def calculate_model_size(stream_wrapper: StreamDiffusionWrapper):

    components = {
        "UNet": stream_wrapper.stream.unet,
        "VAE": stream_wrapper.stream.vae,
        "Text Encoder": stream_wrapper.stream.text_encoder,
    }

    total_params = 0
    for name, model in components.items():
        params = sum(p.numel() for p in model.parameters())
        total_params += params
        print(f"{name} Parameters: {params / 1e6:.2f}M")

    print(f"Total Parameters: {total_params / 1e6:.2f}M")
    return total_params


def run(
    iterations: int = 100,
    model_id_or_path: str = "KBlueLeaf/kohaku-v2.1",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "1girl with brown dog hair, thick glasses, smiling",
    negative_prompt: str = "bad image , bad quality",
    use_lcm_lora: bool = True,
    use_tiny_vae: bool = True,
    width: int = 512,
    height: int = 512,
    warmup: int = 10,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    device_ids: Optional[List[int]] = None,
    use_denoising_batch: bool = True,
    seed: int = 2,
):
    """
    Initializes the StreamDiffusionWrapper.

    Parameters
    ----------
    iterations : int, optional
        The number of iterations to run, by default 100.
    model_id_or_path : str
        The model id or path to load.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str, optional
        The prompt to use, by default "1girl with brown dog hair, thick glasses, smiling".
    negative_prompt : str, optional
        The negative prompt to use, by default "bad image , bad quality".
    use_lcm_lora : bool, optional
        Whether to use LCM-LoRA or not, by default True.
    use_tiny_vae : bool, optional
        Whether to use TinyVAE or not, by default True.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    warmup : int, optional
        The number of warmup steps to perform, by default 10.
    acceleration : Literal["none", "xformers", "tensorrt"], optional
        The acceleration method, by default "tensorrt".
    device_ids : Optional[List[int]], optional
        The device ids to use for DataParallel, by default None.
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    """      
    # stream = StreamDiffusionWrapper(
    #     model_id_or_path=model_id_or_path,
    #     t_index_list=[32, 45],
    #     lora_dict=lora_dict,
    #     mode="img2img",
    #     frame_buffer_size=1,
    #     width=width,
    #     height=height,
    #     warmup=warmup,
    #     acceleration=acceleration,
    #     device_ids=device_ids,
    #     use_lcm_lora=use_lcm_lora,
    #     use_tiny_vae=use_tiny_vae,
    #     enable_similar_image_filter=False,
    #     similar_image_filter_threshold=0.98,
    #     use_denoising_batch=use_denoising_batch,
    #     cfg_type="initialize",  # initialize, full, self , none
    #     seed=seed,
    # )
    base_model = "stabilityai/sd-turbo"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    stream = StreamDiffusionWrapper(
        model_id_or_path=base_model,
        device= device,
        t_index_list=[35, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        use_lcm_lora=False,
        warmup=warmup,
        acceleration=acceleration,
        device_ids=device_ids,
        mode="img2img",
        use_denoising_batch=use_denoising_batch,
        use_tiny_vae=use_tiny_vae,
        cfg_type="none",
        output_type="pil",
        seed=seed,
        enable_similar_image_filter=False,
    )
    if acceleration != "tensorrt":
        calculate_model_size(stream)
    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=1.4,
        delta=0.5,
    )

    downloaded_image = download_image("https://github.com/ddpn08.png").resize(
        (width, height)
    ) 
    image_tensor = stream.preprocess_image(downloaded_image)
    # warmup
    for _ in range(warmup):
        res = stream.stream(image_tensor)

    results = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in tqdm(range(iterations)):
        start.record()

        image_tensor = stream.preprocess_image(downloaded_image)

        # out_tensor =stream(image=image_tensor)
        out_tensor = stream.stream(image_tensor).cpu()
        
        image_res = postprocess_image(out_tensor, output_type="np")

        end.record()
        torch.cuda.synchronize()
        results.append(start.elapsed_time(end))

    print(f"Average time: {sum(results) / len(results)}ms")
    print(f"Average FPS: {1000 / (sum(results) / len(results))}")
    import numpy as np

    fps_arr = 1000 / np.array(results)
    print(f"Max FPS: {np.max(fps_arr)}")
    print(f"Min FPS: {np.min(fps_arr)}")
    print(f"Std: {np.std(fps_arr)}")


if __name__ == "__main__":
    fire.Fire(run)
