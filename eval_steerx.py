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
args = parser.parse_args()

pipe, reward_fn = load_steerx(args.model, device='cuda')

save_path = os.path.join('samples', args.model, str(args.num_particles))

# In eval mode, we disable tqdm
pipe.set_progress_bar_config(disable=True)
torch.set_num_threads(4)
cv2.setNumThreads(0)

if args.model == 'splatflow':  # test t3bench
    from gs_refine.refine import load_refiner, refine_3DGS
    refiner = load_refiner('cuda')
    
    with open('benchmarks/t3bench.txt', 'r') as f:
        text = f.readlines()

    for idx, row in tqdm(enumerate(text)):
        PROMPT = row.strip()

        path = os.path.join(save_path, str(idx))

        if os.path.exists(path):
            continue

        os.makedirs(path, exist_ok=True)

        pipeout = pipe(
            [PROMPT] * args.num_particles,
            reward_fn=reward_fn,
            num_steering=4,
            height=256,
            width=256,
            num_inference_steps=200,
            num_frames=8,
            generator=torch.Generator(device="cuda").manual_seed(42),
        )

        reward_fn.export_scene(pipeout, path)
        refine_3DGS(refiner, Path(path), PROMPT)

elif args.model == 'dimensionx':
    from gs_refine.refine import load_refiner, refine_3DGS
    refiner = load_refiner('cuda')

    fnames = sorted(glob('benchmarks/VBench/*.jpg'))
    
    for idx, fname in tqdm(enumerate(fnames)):
        PROMPT = fname.split('/')[-1].split('.')[0]

        image = load_image(fname)
        path = os.path.join(save_path, PROMPT)

        if os.path.exists(path):
            continue

        os.makedirs(path, exist_ok=True)

        pipeout = pipe(
            [image] * args.num_particles,
            [PROMPT] * args.num_particles,
            reward_fn=reward_fn,
            num_steering=4,
            num_inference_steps=50,
            generator=torch.Generator(device="cuda").manual_seed(42),
            use_dynamic_cfg=True
        )

        export_to_video(pipeout.video[0], os.path.join(path, '_video.mp4'))
        reward_fn.export_scene(pipeout, path)
        refine_3DGS(refiner, Path(path), PROMPT)

elif args.model == 'cog':  # test VBench
    fnames = sorted(glob('benchmarks/VBench/*.jpg'))
    for fname in tqdm(fnames):
        PROMPT = fname.split('/')[-1].split('.')[0]
        negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

        image = load_image(fname)
        path = os.path.join(save_path, PROMPT)

        if os.path.exists(path):
            continue

        os.makedirs(path, exist_ok=True)

        pipeout = pipe(
            [image] * args.num_particles,
            [PROMPT] * args.num_particles,
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

        export_to_video(pipeout.video[0], os.path.join(path, '_video.mp4'))
        reward_fn.export_scene(pipeout, path)

else:  # test penguin
    df = pd.read_csv('benchmarks/penguin_camera.csv', index_col=0)

    for idx, row in tqdm(df.iterrows()):
        PROMPT = row.to_list()[0]

        path = os.path.join(save_path, str(idx))
        if os.path.exists(path):
            continue

        os.makedirs(path, exist_ok=True)

        pipeout = pipe(
            [PROMPT] * args.num_particles,
            reward_fn=reward_fn,
            num_steering=2,
            height=480,
            width=480,
            num_frames=49,
            num_inference_steps=50,
            generator=torch.Generator(device="cuda").manual_seed(42)
        )

        export_to_video(pipeout.video[0], os.path.join(path, '_video.mp4'))
        reward_fn.export_scene(pipeout, path)
