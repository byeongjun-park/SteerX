"""
Following SDS++ in Director3D, we refine initialized 3DGS.
"""

import os
import torch
import numpy as np
from pathlib import Path
from gs_refine.sds_pp_refiner import GSRefinerSDSPlusPlus
from gs_refine.vis_utils import *
from tqdm import tqdm


def load_refiner(device):
    refiner = GSRefinerSDSPlusPlus(
        sd_model_key='stabilityai/stable-diffusion-2-1-base',
        num_views=1,
        img_size=512,
        guidance_scale=7.5,
        min_step_percent=0.02,
        max_step_percent=0.5,
        num_densifications=4,
        lr_scale=0.25,
        lrs={'xyz': 2e-4, 'features': 1e-2, 'opacity': 5e-2, 'scales': 1e-3, 'rotations': 1e-2, 'embeddings': 1e-2},
        use_lods=True,
        lambda_latent_sds=1,
        lambda_image_sds=0.1,
        lambda_image_variation=0.001,
        opacity_threshold=0.01,
        text_templete="$text$",
        negative_text_templete='unclear. noisy. point cloud. low-res. low-quality. low-resolution. unrealistic.',
        total_iterations=1000,
    ).to(device)

    return refiner

def refine_3DGS(refiner, path, text):
    device = 'cuda'

    init_3DGS = load_ply_for_gaussians(path / "scene.ply", device=device)
    Ks = torch.load(path / 'Ks.pt', weights_only=False).to(device)
    viewmats = torch.load(path / 'viewmats.pt', weights_only=False).to(device)

    c2ws = torch.linalg.inv(viewmats)[:, :3]
    cameras = torch.cat([Ks, c2ws], dim=-1).unsqueeze(0).to(device)

    gs_xyz = init_3DGS[0]
    mean_xyz = gs_xyz.mean(dim=1, keepdim=True)
    gs_xyz = gs_xyz - mean_xyz
    cameras[..., -1] = cameras[..., -1] - mean_xyz

    init_3DGS = (gs_xyz, *init_3DGS[1:])
    

    # Refine init_3DGS
    refined_gaussians = refiner.refine_gaussians(init_3DGS, text, dense_cameras=cameras)

    # Save the refined 3DGS
    output_path = os.path.join(path, f'refined')
    list_refined_gaussians = [p[0] for p in refined_gaussians]
    export_ply_for_gaussians(output_path, list_refined_gaussians, mode='export_for_splatflow')

    # Render multi-view images and a video of the refined 3DGS
    render_fn = lambda cameras, h, w: refiner.renderer(cameras, refined_gaussians, h=h, w=w, bg=None)[:2]
    export_mv(render_fn, os.path.join(path, f'refined_render_img'), cameras)
    export_video(render_fn, path, f"refined", cameras, device=device)