import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'monst3r'))

import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch import nn, Tensor
from einops import rearrange, repeat
from copy import deepcopy
import numpy as np

# Load Pytorch3D
from pytorch3d.structures import Pointclouds
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)

from featup.util import norm
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import loss_of_one_batch
from dust3r.image_pairs import make_pairs
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode, init_im_poses
from torchvision.utils import save_image

class GeometryReward(nn.Module):
    def __init__(self, monst3r_ckpt):
        super().__init__()

        model = AsymmetricCroCo3DStereo.from_pretrained(monst3r_ckpt)
        self.model = model.eval()
        self.optimizer = GlobalAlignerMode.PointCloudOptimizer  
        self.upsampler = torch.hub.load('mhamilton723/FeatUp', 'dino16', use_norm=True).eval()
        raster_settings = PointsRasterizationSettings(
            image_size=224,
            radius=0.005, 
            points_per_pixel=10,
            bin_size=0,
        )

        self.renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=None, raster_settings=raster_settings),
            compositor=AlphaCompositor()
        )

    def get_3D_scene_from_frames(self, output, device):
        scene = global_aligner(
                to_cpu(output),
                device=device,
                mode=self.optimizer, 
                verbose=False, 
                shared_focal=True,
                temporal_smoothing_weight=0.01,
                translation_weight=1.0,
                flow_loss_weight=0.01,
                flow_loss_start_epoch=0.1, 
                flow_loss_thre=25, 
                use_self_mask=True,
                sam2_mask_refine=True,
                num_total_iter=300,
                empty_cache=False, 
                batchify=True
            )

        with torch.enable_grad():
            _ = scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=0.01)

        return scene

    @torch.no_grad()
    def export_scene(self, pipeout, path):
        with open(os.path.join(path, 'reward.txt'), 'w') as f:
            f.write(str(pipeout.reward.item()))

        scene = pipeout.scene

        scene.clean_pointcloud()
        poses = scene.save_tum_poses(f'{path}/pred_traj.txt')
        K = scene.save_intrinsics(f'{path}/pred_intrinsics.txt')
        depth_maps = scene.save_depth_maps(path)
        conf = scene.save_conf_maps(path)
        scene.save_init_conf_maps(path)
        rgbs = scene.save_rgb_imgs(path)
        masks = scene.save_dynamic_masks(path)

        return scene

    @torch.no_grad()
    def forward(self, video, size=224):
        n_frames = len(video)  # 4x + 1 or 6x + 1
        original_video = F.interpolate(video, size=(size, size), mode='bicubic').clamp(0, 1)

        src_indices = np.linspace(0, n_frames - 1, n_frames // 2 + 1, dtype=np.int64)
        tgt_indcies = np.delete(np.arange(n_frames), src_indices)

        with torch.amp.autocast('cuda', enabled=True):
            src_feats = self.upsampler(norm(original_video[src_indices]))
            tgt_feats = self.upsampler(norm(original_video[tgt_indcies]))

        src_feats = src_feats.to(torch.float32)
        tgt_feats = tgt_feats.to(torch.float32)

        video = original_video * 2 - 1
        frames = []
        for frame in video:
            idx = len(frames)
            frames.append(
                {
                    'img': frame[None].to(torch.float32),
                    'true_shape': torch.tensor([[size, size]]).long(),
                    'idx': idx,
                    'instance': str(idx),
                    'mask': ~(frame[None].sum(1) <= -0.99),
                    'dynamic_mask': torch.zeros_like(frame[None])
                }
            )

        pairs = make_pairs(frames, scene_graph=f'swin_4_steerx', prefilter=None, symmetrize=True) # stride 4, window size 4, non-cyclic
        output = loss_of_one_batch(collate_with_cat(pairs), self.model, None, video.device)
        scene = self.get_3D_scene_from_frames(output, device='cuda')

        Ks = scene.get_intrinsics()
        viewmats = torch.linalg.inv(scene.get_im_poses())

        all_pts3d = torch.stack(scene.get_pts3d(), dim=0)  # v h w xyz
        all_masks = torch.stack(scene.sam2_dynamic_masks, dim=0)  # v h w
        
        src_masks = rearrange(all_masks[src_indices], 'v h w -> (v h w)')
        tgt_masks = rearrange(all_masks[tgt_indcies], 'v h w -> (v h w)')

        if (~src_masks).sum() == 0:
            src_pts3d = rearrange(all_pts3d[src_indices], 'v h w c -> (v h w) c')
            src_feats = rearrange(src_feats, 'v c h w -> (v h w) c')
            tgt_masks = torch.zeros_like(tgt_masks, dtype=bool)
        else:
            src_pts3d = rearrange(all_pts3d[src_indices], 'v h w c -> (v h w) c')[~src_masks]
            src_feats = rearrange(src_feats, 'v c h w -> (v h w) c')[~src_masks]

        tgt_feats = rearrange(tgt_feats, 'v c h w -> (v h w) c')

        tgt_cameras = cameras_from_opencv_projection(
            R=viewmats[tgt_indcies, :3, :3],
            tvec=viewmats[tgt_indcies, :3, -1],
            camera_matrix=Ks[tgt_indcies],
            image_size=repeat(torch.tensor([size, size], device=Ks.device), 'i -> v i', v=len(tgt_indcies))
        )

        bg_feat = [-10000] * src_feats.shape[-1]
        rendered_feats = []
        for i in range(len(tgt_cameras)):
            current_pc = Pointclouds(points=[src_pts3d], features=[src_feats])
            rendered_feat = self.renderer(current_pc, cameras=tgt_cameras[i], background_color=bg_feat)
            rendered_feats.append(rendered_feat)

        rendered_feats = torch.cat(rendered_feats, dim=0)
        rendered_feats = rearrange(rendered_feats, 'v h w c -> (v h w) c')

        reward = F.cosine_similarity(tgt_feats[~tgt_masks], rendered_feats[~tgt_masks]).clamp(min=0).mean()

        return reward, scene
