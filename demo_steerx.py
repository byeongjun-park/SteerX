import os

os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["RAYON_NUM_THREADS"] = "4"


from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
import pandas as pd
import torch
from diffusers.utils import load_image, export_to_video
from steerx_diffusers import load_steerx
import cv2
from pathlib import Path

os.environ['PYTHONPATH'] = os.getcwd()


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

parser = ArgumentParser()
parser.add_argument('--model', type=str, default='splatflow', choices=['splatflow', 'dimensionx', 'cog', 'mochi', 'hunyuan'])
parser.add_argument('--num_particles', type=int, default=4)  # batch size
parser.add_argument('--prompt', type=str)
parser.add_argument('--img_path', type=str) 
args = parser.parse_args()

pipe, reward_fn = load_steerx(args.model, device='cuda')

save_path = os.path.join('samples', args.model, str(args.num_particles), args.prompt[:50])
os.makedirs(save_path, exist_ok=True)
with open(os.path.join(save_path, 'prompt'),  'w') as f:
    f.write(args.prompt)

# In eval mode, we disable tqdm
pipe.set_progress_bar_config(disable=True)
torch.set_num_threads(4)
cv2.setNumThreads(0)

if args.model == 'splatflow':  # test t3bench
    from gs_refine.refine import load_refiner, refine_3DGS
    refiner = load_refiner('cuda')

    pipeout = pipe(
        [args.prompt] * args.num_particles,
        reward_fn=reward_fn,
        num_steering=4,
        height=256,
        width=256,
        num_inference_steps=200,
        num_frames=8,
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    reward_fn.export_scene(pipeout, save_path)
    refine_3DGS(refiner, Path(save_path), args.prompt)

elif args.model == 'dimensionx':
    from gs_refine.refine import load_refiner, refine_3DGS
    refiner = load_refiner('cuda')

    image = load_image(args.img_path)

    pipeout = pipe(
        [image] * args.num_particles,
        [args.prompt] * args.num_particles,
        reward_fn=reward_fn,
        num_steering=4,
        num_inference_steps=50,
        generator=torch.Generator(device="cuda").manual_seed(42),
        use_dynamic_cfg=True
    )

    export_to_video(pipeout.video[0], os.path.join(save_path, '_video.mp4'))
    reward_fn.export_scene(pipeout, save_path)
    refine_3DGS(refiner, Path(save_path), args.prompt)

elif args.model == 'cog':  # test VBench
    image = load_image(args.img_path)
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    pipeout = pipe(
        [image] * args.num_particles,
        [args.prompt] * args.num_particles,
        negative_prompt=negative_prompt,
        reward_fn=reward_fn,
        num_steering=2,
        height=768,
        width=768,
        num_inference_steps=50,
        num_frames=49,
        generator=torch.Generator(device="cuda").manual_seed(42),
        use_dynamic_cfg=True
    )

    export_to_video(pipeout.video[0], os.path.join(save_path, '_video.mp4'))
    reward_fn.export_scene(pipeout, save_path)

else:  # test penguin
    pipeout = pipe(
        [args.prompt] * args.num_particles,
        reward_fn=reward_fn,
        num_steering=2,
        height=480,
        width=480,
        num_frames=49,
        num_inference_steps=50,
        generator=torch.Generator(device="cuda").manual_seed(42)
    )

    export_to_video(pipeout.video[0], os.path.join(save_path, '_video.mp4'))
    reward_fn.export_scene(pipeout, save_path)
