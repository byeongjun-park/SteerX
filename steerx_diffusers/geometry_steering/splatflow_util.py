import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from gsplat.rendering import rasterization
from torchvision.utils import save_image
import numpy as np
from featup.util import norm
from plyfile import PlyData, PlyElement
from steerx_diffusers.splatflow.gs_decoder_architecture import GSDecoder

class GaussianRenderer(nn.Module):
    def __init__(self, znear=0.05, zfar=20.0):
        super().__init__()
        self.znear = znear
        self.zfar = zfar

        self.register_buffer("bg_color", torch.zeros((1, 1), dtype=torch.float32))

    def forward(self, Ks, viewmats, gaussians, image_shape):
        im_height, im_width = image_shape


        bg_color = self.bg_color.repeat(Ks.shape[0], gaussians['rgb'].shape[-1]).to(Ks.device)
        out_img, _, _ = rasterization(
            means=gaussians['means'],
            quats=gaussians['rotation'],
            scales=gaussians['scale'],
            opacities=gaussians['opacity'],
            colors=gaussians['rgb'],
            viewmats=viewmats,
            Ks=Ks,
            width=im_width,
            height=im_height,
            near_plane=self.znear,
            far_plane=self.zfar,
            backgrounds=bg_color,
            sh_degree=None,
            radius_clip=0.0,
        )

        out_img = rearrange(out_img, 'v h w c -> v c h w')

        return out_img

class GeometryReward(nn.Module):
    def __init__(self, decoder_ckpt):
        super().__init__()

        decoder = GSDecoder()
        decoder.load_state_dict(torch.load(decoder_ckpt, map_location="cpu", weights_only=False))

        self.model = decoder.eval()
        self.renderer = GaussianRenderer()
        self.upsampler = torch.hub.load('mhamilton723/FeatUp', 'dino16', use_norm=True).eval()

    @torch.no_grad()
    def export_scene(self, pipeout, path):
        with open(os.path.join(path, 'reward.txt'), 'w') as f:
            f.write(str(pipeout.reward.item()))

        gaussians, Ks, viewmats = pipeout.scene

        torch.save(Ks, os.path.join(path, 'Ks.pt'))
        torch.save(viewmats, os.path.join(path, 'viewmats.pt'))

        rendered_video = self.renderer(Ks, viewmats, gaussians, image_shape=(224, 224)).clamp(min=0, max=1)

        for i, (sample_img, render_img) in enumerate(zip(pipeout.mv_imgs, rendered_video)):
            sample_img.save(os.path.join(path, f'sample_img_{i}.png'))
            save_image(render_img, os.path.join(path, f'render_img{i}.png'))

        # adust values
        gaussians['rgb'] = (gaussians['rgb'] - 0.5) / 0.28209479177387814
        gaussians['opacity'] = torch.log(gaussians['opacity'] / (1 - gaussians['opacity']))  # inverse sigmoid
        gaussians['scale'] = torch.log(gaussians['scale'] + 1e-8)

        xyzs = gaussians['means'].detach().cpu().numpy()
        f_dc = gaussians['rgb'].detach().cpu().numpy()
        opacities = gaussians['opacity'].unsqueeze(-1).detach().cpu().numpy()
        scales = gaussians['scale'].detach().cpu().numpy()
        rotations = gaussians['rotation'].detach().cpu().numpy()

        l = ["x", "y", "z"]  # noqa: E741
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append("f_dc_{}".format(i))
        l.append("opacity")
        for i in range(scales.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(rotations.shape[1]):
            l.append("rot_{}".format(i))

        dtype_full = [(attribute, "f4") for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(os.path.join(path, 'scene.ply'))

        return path

    @torch.no_grad()
    def forward(self, latents, video):
        h, w = video.shape[-2:]

        gaussians, Ks, viewmats = self.model(latents)
        gaussians = {k: rearrange(v, 'v h w ... -> (v h w)... ').float() for k, v in gaussians.items()}

        rendered_video = self.renderer(Ks, viewmats, gaussians, image_shape=(h, w)).clamp(min=0, max=1)

        with torch.amp.autocast('cuda', enabled=True):
            dino_feat = self.upsampler(norm(video))
            rendered_dino_feat = self.upsampler(norm(rendered_video))

            reward = torch.nn.functional.cosine_similarity(dino_feat, rendered_dino_feat).mean().clamp(min=0)

        return reward, (gaussians, Ks, viewmats)
