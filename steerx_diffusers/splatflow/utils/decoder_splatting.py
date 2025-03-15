import torch
from torch import nn
from einops import einsum, rearrange, repeat
from scipy.spatial.transform import Rotation as R


def meshgrid(spatial_shape, normalized=True, indexing="ij", device=None):
    """Create evenly spaced position coordinates for self.spatial_shape with values in [v_min, v_max].
    :param v_min: minimum coordinate value per dimension.
    :param v_max: maximum coordinate value per dimension.
    :return: position coordinates tensor of shape (*shape, len(shape)).
    """
    if normalized:
        axis_coords = [torch.linspace(-1.0, 1.0, steps=s, device=device) for s in spatial_shape]
    else:
        axis_coords = [torch.linspace(0, s - 1, steps=s, device=device) for s in spatial_shape]

    grid_coords = torch.meshgrid(*axis_coords, indexing=indexing)

    return torch.stack(grid_coords, dim=-1)


class DecoderSplatting(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("background_color", torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32), persistent=False)
        self.act_scale = nn.Softplus()
        self.act_rgb = nn.Softplus()

    def get_scale_multiplier(self, intrinsics):
        pixel_size = torch.ones((2,), dtype=torch.float32, device=intrinsics.device)
        xy_multipliers = einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    def forward(self, raw_gaussians, extrinsics, intrinsics, near=0.05, far=20):
        b, v, c, h, w = raw_gaussians.shape
        extrinsics = repeat(extrinsics, "b v i j -> b v () () i j")
        intrinsics = repeat(intrinsics, "b v i j -> b v () () i j")
        raw_gaussians = rearrange(raw_gaussians, "b v c h w -> b v h w c")

        rgb, disp, opacity, scales, rotations, xy_offset = raw_gaussians.split((3, 1, 1, 3, 4, 2), dim=-1)

        # calculate xy_offset and origin/direction for each view.
        pixel_coords = meshgrid((w, h), normalized=False, indexing="xy", device=raw_gaussians.device)
        pixel_coords = repeat(pixel_coords, "h w c -> b v h w c", b=b, v=v)

        coordinates = pixel_coords + (xy_offset.sigmoid() - 0.5)
        coordinates = torch.cat([coordinates, torch.ones_like(coordinates[..., :1])], dim=-1)

        directions = einsum(intrinsics.inverse(), coordinates, "... i j, ... j -> ... i")
        directions = directions / directions.norm(dim=-1, keepdim=True)
        directions = torch.cat([directions, torch.zeros_like(directions[..., :1])], dim=-1)
        directions = einsum(extrinsics, directions, "... i j, ... j -> ... i")
        origins = extrinsics[..., -1].broadcast_to(directions.shape)

        # calculate depth from disparity
        depths = 1.0 / (disp.sigmoid() * (1.0 / near - 1.0 / far) + 1.0 / far)

        # calculate all parameters of gaussian splats
        means = origins + directions * depths

        multiplier = self.get_scale_multiplier(intrinsics)
        scales = self.act_scale(scales) * multiplier[..., None]

        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)

        flat_rotations = rearrange(rotations.detach().cpu().numpy(), 'b v h w q -> (b v h w) q')
        cam_rotation_matrix = R.from_quat(flat_rotations).as_matrix()

        source_rotations = repeat(extrinsics[..., :3, :3], 'b v () () i j -> (b v h w) i j', h=h, w=w)
        world_rotation_matrix = source_rotations.detach().cpu().numpy() @ cam_rotation_matrix
        world_rotations = R.from_matrix(world_rotation_matrix).as_quat(scalar_first=True)
        world_rotations = torch.from_numpy(world_rotations).to(source_rotations.device)
        world_rotations = rearrange(world_rotations, '(b v h w) q -> b v h w q', b=b, v=v, h=h, w=w)

        opacity = opacity.sigmoid()
        rgb = self.act_rgb(rgb)

        gaussians = dict(
            means=means[0],
            rgb=rgb[0],
            opacity=opacity[0].squeeze(-1),
            scale=scales[0],
            rotation=world_rotations[0]
        )
        
        return gaussians