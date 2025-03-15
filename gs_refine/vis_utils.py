# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tqdm
import imageio
import numpy as np
from io import BytesIO
from einops import repeat, rearrange

import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from gs_refine.camera_utils import sample_from_dense_cameras


def export_ply_for_gaussians(path, gaussians, mode, export_splat=False):
    if isinstance(gaussians, dict):
        xyz, features, opacity, scales, rotations = gaussians.values()
    else:
        xyz, features, opacity, scales, rotations = gaussians

    means3D = xyz.contiguous().float()
    opacity = opacity.contiguous().float()
    scales = scales.contiguous().float()
    rotations = rotations.contiguous().float()
    shs = features.contiguous().float()  # [N, 1, 3]

    SH_C0 = 0.28209479177387814
    rotations, shs = adjust_gaussians(rotations, shs, SH_C0, mode=mode)

    opacity = torch.log(opacity / (1 - opacity))  # inverse sigmoid
    scales = torch.log(scales + 1e-8)

    xyzs = means3D.detach().cpu().numpy()
    f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    rotations = rotations.detach().cpu().numpy()

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

    PlyData([el]).write(path + '.ply')

    if export_splat:
        plydata = PlyData([el])

        vert = plydata["vertex"]
        sorted_indices = np.argsort(
            -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"]) / (1 + np.exp(-vert["opacity"]))
        )
        buffer = BytesIO()
        for idx in sorted_indices:
            v = plydata["vertex"][idx]
            position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
            scales = np.exp(
                np.array(
                    [v["scale_0"], v["scale_1"], v["scale_2"]],
                    dtype=np.float32,
                )
            )
            rot = np.array(
                [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                dtype=np.float32,
            )
            color = np.array(
                [
                    0.5 + SH_C0 * v["f_dc_0"],
                    0.5 + SH_C0 * v["f_dc_1"],
                    0.5 + SH_C0 * v["f_dc_2"],
                    1 / (1 + np.exp(-v["opacity"])),
                ]
            )
            buffer.write(position.tobytes())
            buffer.write(scales.tobytes())
            buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
            buffer.write(((rot / np.linalg.norm(rot)) * 128 + 128).clip(0, 255).astype(np.uint8).tobytes())

        with open(path + '.splat', "wb") as f:
            f.write(buffer.getvalue())


def load_ply_for_gaussians(path, device="cpu"):
    plydata = PlyData.read(path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    print("Number of points at loading : ", xyz.shape[0])

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = torch.tensor(xyz, dtype=torch.float, device=device)[None]
    features = torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2)[None]
    opacity = torch.tensor(opacities, dtype=torch.float, device=device)[None]
    scales = torch.tensor(scales, dtype=torch.float, device=device)[None]
    rotations = torch.tensor(rots, dtype=torch.float, device=device)[None]

    opacity = torch.sigmoid(opacity)
    scales = torch.exp(scales)

    SH_C0 = 0.28209479177387814
    rotations, features = adjust_gaussians(rotations, features, SH_C0, mode='load_for_sds')

    return xyz, features, opacity, scales, rotations

def adjust_gaussians(rotations, shs, SH_C0, mode):
    if mode=='load_for_sds':  # load: convert wxyz --> xyzw (rotation), convert shs to precomputed color
        rotations = rotations[:, :, [1, 2, 3, 0]]
        shs = 0.5 + shs * SH_C0
    elif mode == 'export_for_splatflow':  # export: convert xyzw --> wxyz (rotation), convert precomputed color to shs
        rotations = rotations[:, [3, 0, 1, 2]]
        shs = (shs - 0.5) / SH_C0
    elif mode == 'export_for_mvdust3rp':
        shs = (shs - 0.5) / SH_C0
    else:
        raise ValueError('Mode must be one of load_for_sds, export_for_splatflow, or export_for_mvdust3rp')

    return rotations, shs


@torch.no_grad()
def export_video(render_fn, save_path, name, dense_cameras, fps=60, num_frames=720, size=512, device="cuda:0"):
    images = []

    for i in tqdm.trange(num_frames, desc="Rendering video..."):
        t = torch.full((1, 1), fill_value=i / num_frames, device=device)
        camera = sample_from_dense_cameras(dense_cameras, t, noise_strengths=[0, 0])
        image = render_fn(camera, size, size)[0]
        images.append(process_image(image.reshape(3, size, size)))

    imageio.mimwrite(os.path.join(save_path, f"{name}.mp4"), images, fps=fps, quality=8, macro_block_size=1)

def process_image(image):
    return image.permute(1, 2, 0).detach().cpu().mul(1 / 2).add(1 / 2).clamp(0, 1).mul(255).numpy().astype(np.uint8)

@torch.no_grad()
def export_mv(render_fn, save_path, dense_cameras, size=256):
    num_views = dense_cameras.shape[1]
    imgs = []
    for i in tqdm.trange(num_views, desc="Rendering images..."):
        image, depth = render_fn(dense_cameras[:, i].unsqueeze(1), size, size)
        image = image.reshape(3, size, size).clamp(-1, 1).add(1).mul(1 / 2)
        disp = 1 / depth
        disp = (disp - disp.min()) / (disp.max() - disp.min())
        disp = disp.reshape(1, size, size)
        imgs.append(image)
        save_image(image, save_path + f'_{i}.png')
        save_image(disp, save_path.replace('img', 'depth') + f'_{i}.png')

    cmap = plt.get_cmap("hsv")
    num_rows = 2
    num_cols = num_views // num_rows
    figsize = (num_cols * 2, num_rows * 2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()
    for i in range(num_rows * num_cols):
        if i < num_views:
            axs[i].imshow((imgs[i].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8))
            for s in ["bottom", "top", "left", "right"]:
                axs[i].spines[s].set_color(cmap(i / (num_views)))
                axs[i].spines[s].set_linewidth(5)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        else:
            axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path + 's.pdf', transparent=True)