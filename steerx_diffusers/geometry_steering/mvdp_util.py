import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mvdust3r'))

import torch
from torchvision.transforms import transforms
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from gsplat.rendering import rasterization
from torchvision.utils import save_image
import numpy as np
import time
from copy import deepcopy
from featup.util import norm
from plyfile import PlyData, PlyElement
import trimesh
from scipy.spatial.transform import Rotation
from dust3r.model import AsymmetricCroCo3DStereoMultiView
from dust3r.dummy_io import get_local_path
from dust3r.losses import calibrate_camera_pnpransac, estimate_focal_knowing_depth
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, cat_meshes, OPENGL, pts3d_to_trimesh

# Load Pytorch3D
from pytorch3d.structures import Pointclouds
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)



class GaussianRenderer(nn.Module):
    def __init__(self, znear=0.01, zfar=100.0):
        super().__init__()
        self.znear = znear
        self.zfar = zfar

        self.register_buffer("bg_color", torch.zeros((1, 1), dtype=torch.float32))

    def forward(self, Ks, viewmats, gaussians, image_shape): # we assume the input rgb should be -1~1 if it is not sh
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

def sort_frames(imgs):
    # specific case for 8 images following original repo in MV-DUSt3R.
    imgs[2], imgs[6] = deepcopy(imgs[6]), deepcopy(imgs[2])

    return imgs

def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.1,
                                 cam_color=None, as_pointcloud=True, transparent_cams=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    scene.export(file_obj=outdir)

def get_3D_model_from_scene(output, outdir=None, min_conf_thr=3):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """

    with torch.no_grad():
        v, h, w = output['rgb'].shape[0:3]
        rgbimg = output['rgb']

        pts3d = output['pts3d']
        conf = output['conf']
        conf_sorted = conf.reshape(-1).sort()[0]
        conf_thres = conf_sorted[int(conf_sorted.shape[0] * float(min_conf_thr) * 0.01)]
        msk = conf >= conf_thres

        # calculate focus:
        conf_first = conf[0].reshape(-1)  # [bs, H * W]
        conf_sorted = conf_first.sort()[0]  # [bs, h * w]
        conf_thres = conf_sorted[int(conf_first.shape[0] * 0.03)]
        valid_first = (conf_first >= conf_thres)  # & valids[0].reshape(bs, -1)
        valid_first = valid_first.reshape(h, w)

        focals = estimate_focal_knowing_depth(pts3d[0][None].cuda(), valid_first[None].cuda()).cpu().item()

        intrinsics = torch.eye(3, )
        intrinsics[0, 0] = focals
        intrinsics[1, 1] = focals
        intrinsics[0, 2] = w / 2
        intrinsics[1, 2] = h / 2
        intrinsics = intrinsics.cuda()

        focals = torch.Tensor([focals]).reshape(1, ).repeat(len(rgbimg))

        y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float().cuda()  # [H, W, 2]

        c2ws = []
        for (pr_pt, valid) in zip(pts3d, msk):
            c2ws_i = calibrate_camera_pnpransac(pr_pt.cuda().flatten(0, 1)[None], pixel_coords.flatten(0, 1)[None],
                                                valid.cuda().flatten(0, 1)[None], intrinsics[None])
            c2ws.append(c2ws_i[0])

        cams2world = torch.stack(c2ws, dim=0).cpu()  # [N, 4, 4]
        focals = to_numpy(focals)

        pts3d = to_numpy(pts3d)
        msk = to_numpy(msk)

        sorted_c2w = sort_frames(deepcopy(cams2world.to(intrinsics.device)))
        Ks = repeat(intrinsics, 'i j -> v i j', v=v)

        if outdir is not None:
            _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world)

    return Ks, sorted_c2w.inverse()


class GeometryReward(nn.Module):
    def __init__(self, mvdust3rp_ckpt):
        super().__init__()

        model = AsymmetricCroCo3DStereoMultiView(pos_embed='RoPE100', img_size=(224, 224), head_type='linear',
                                                output_mode='pts3d', depth_mode=('exp', -np.inf, np.inf),
                                                conf_mode=('exp', 1, 1e9), enc_embed_dim=1024, enc_depth=24,
                                                enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12,
                                                GS=True, sh_degree=0, pts_head_config={'skip': True}, m_ref_flag=True,
                                                n_ref=4)

        model_loaded = AsymmetricCroCo3DStereoMultiView.from_pretrained(get_local_path(mvdust3rp_ckpt))
        state_dict_loaded = model_loaded.state_dict()
        model.load_state_dict(state_dict_loaded, strict=True)

        self.model = model.eval()
        self.renderer = GaussianRenderer()
        self.upsampler = torch.hub.load('mhamilton723/FeatUp', 'dino16', use_norm=True).eval()

    @torch.no_grad()
    def export_scene(self, pipeout, path):
        with open(os.path.join(path, 'reward.txt'), 'w') as f:
            f.write(str(pipeout.reward.item()))

        scene = pipeout.scene

        Ks, viewmats = get_3D_model_from_scene(scene, outdir=os.path.join(path, 'scene.glb'))

        torch.save(Ks, os.path.join(path, 'Ks.pt'))
        torch.save(viewmats, os.path.join(path, 'viewmats.pt'))

        conf = scene['conf']
        conf_sorted = conf.reshape(-1).sort()[0]
        conf_thres = conf_sorted[int(conf_sorted.shape[0] * 0.03)]
        msk = conf >= conf_thres

        gaussians = {
            'means': scene['pts3d'],
            'rgb': scene['rgb'],
            'opacity': scene['opacity'].squeeze(-1),
            'scale': scene['scale'].clamp(min=1e-4, max=0.02),
            'rotation': scene['rotation']
        }

        gaussians = {k: v[msk].float() for k, v in gaussians.items()}

        sort_img = sort_frames(deepcopy(scene['rgb']))
        for i, sample_img in enumerate(sort_img):
            render_img = self.renderer(Ks[i][None], viewmats[i][None], gaussians, image_shape=(224, 224)).clamp(min=0, max=1)
            save_image(sample_img.permute(2, 0, 1), os.path.join(path, f'sample_img_{i}.png'))
            save_image(render_img[0], os.path.join(path, f'render_img_{i}.png'))


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
    def forward(self, video, size=224):
        sub_indices = np.linspace(0, len(video) - 1, 8, dtype=np.int64)
        original_video = F.interpolate(video[sub_indices], size=(size, size), mode='bicubic').clamp(0,1)

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
                }
            )

        frames = sort_frames(frames)
        pred1, pred2s = self.model(frames[0], frames[1:])

        # Collect output
        total_parameters = {
            'pts3d': torch.stack([pred1['pts3d'][0]] + [x['pts3d_in_other_view'][0] for x in pred2s], 0),
            'conf': torch.stack([pred1['conf'][0]] + [x['conf'][0] for x in pred2s], 0),
            'rgb': torch.cat([(img['img'].permute(0, 2, 3, 1) + 1) / 2 for img in frames], 0),
            'opacity': torch.stack([pred1['opacity'][0]] + [x['opacity'][0] for x in pred2s], 0),
            'scale': torch.stack([pred1['scale'][0]] + [x['scale'][0] for x in pred2s], 0),
            'rotation': torch.stack([pred1['rotation'][0]] + [x['rotation'][0] for x in pred2s], 0),
        }

        Ks, viewmats = get_3D_model_from_scene(total_parameters)

        conf = total_parameters['conf']
        conf_sorted = conf.reshape(-1).sort()[0]
        conf_thres = conf_sorted[int(conf_sorted.shape[0] * 0.03)]
        msk = conf >= conf_thres

        gaussians = {
            'means': total_parameters['pts3d'],
            'rgb': total_parameters['rgb'],
            'opacity': total_parameters['opacity'].squeeze(-1),
            'scale': total_parameters['scale'].clamp(min=1e-4, max=0.02),
            'rotation': total_parameters['rotation']
        }

        gaussians = {k: v[msk].float() for k, v in gaussians.items()}
        rendered_video = self.renderer(Ks, viewmats, gaussians, image_shape=(size, size))
        with torch.amp.autocast('cuda', enabled=True):
            rendered_dino_feat = self.upsampler(norm(rendered_video.clamp(min=0, max=1)))
            dino_feat = self.upsampler(norm(original_video))

            reward = F.cosine_similarity(dino_feat, rendered_dino_feat).clamp(min=0)
        
        reward = reward[msk].mean()

        return reward, total_parameters
