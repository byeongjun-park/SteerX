import math
import torch
from einops import repeat

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(M: torch.Tensor) -> torch.Tensor:
    """
    Matrix-to-quaternion conversion method. Equation taken from
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    Args:
        M: rotation matrices, (... x 3 x 3)
    Returns:
        q: quaternion of shape (... x 4)
    """
    prefix_shape = M.shape[:-2]
    Ms = M.reshape(-1, 3, 3)

    trs = 1 + Ms[:, 0, 0] + Ms[:, 1, 1] + Ms[:, 2, 2]

    Qs = []

    for i in range(Ms.shape[0]):
        M = Ms[i]
        tr = trs[i]
        if tr > 0:
            r = torch.sqrt(tr) / 2.0
            x = (M[2, 1] - M[1, 2]) / (4 * r)
            y = (M[0, 2] - M[2, 0]) / (4 * r)
            z = (M[1, 0] - M[0, 1]) / (4 * r)
        elif (M[0, 0] > M[1, 1]) and (M[0, 0] > M[2, 2]):
            S = torch.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2  # S=4*qx
            r = (M[2, 1] - M[1, 2]) / S
            x = 0.25 * S
            y = (M[0, 1] + M[1, 0]) / S
            z = (M[0, 2] + M[2, 0]) / S
        elif M[1, 1] > M[2, 2]:
            S = torch.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2  # S=4*qy
            r = (M[0, 2] - M[2, 0]) / S
            x = (M[0, 1] + M[1, 0]) / S
            y = 0.25 * S
            z = (M[1, 2] + M[2, 1]) / S
        else:
            S = torch.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2  # S=4*qz
            r = (M[1, 0] - M[0, 1]) / S
            x = (M[0, 2] + M[2, 0]) / S
            y = (M[1, 2] + M[2, 1]) / S
            z = 0.25 * S
        Q = torch.stack([r, x, y, z], dim=-1)
        Qs += [Q]

    return torch.stack(Qs, dim=0).reshape(*prefix_shape, 4)


@torch.amp.autocast("cuda", enabled=False)
def quaternion_slerp(q0, q1, fraction, spin: int = 0, shortestpath: bool = True):
    """Return spherical linear interpolation between two quaternions.
    Args:
        quat0: first quaternion
        quat1: second quaternion
        fraction: how much to interpolate between quat0 vs quat1 (if 0, closer to quat0; if 1, closer to quat1)
        spin: how much of an additional spin to place on the interpolation
        shortestpath: whether to return the short or long path to rotation
    """
    d = (q0 * q1).sum(-1)
    if shortestpath:
        # invert rotation
        d[d < 0.0] = -d[d < 0.0]
        q1[d < 0.0] = q1[d < 0.0]

    d = d.clamp(0, 1.0).unsqueeze(-1)

    angle = torch.acos(d) + spin * math.pi
    isin = 1.0 / (torch.sin(angle) + 1e-10)
    q0_ = q0 * torch.sin((1.0 - fraction) * angle) * isin
    q1_ = q1 * torch.sin(fraction * angle) * isin

    q = q0_ + q1_
    q[angle.squeeze(-1) < 1e-5, :] = q0[angle.squeeze(-1) < 1e-5, :]

    return q


def sample_from_two_pose(pose_a, pose_b, fraction, noise_strengths):
    """
    Args:
        pose_a: first pose
        pose_b: second pose
        fraction
    """

    quat_a = matrix_to_quaternion(pose_a[..., :3, :3])
    quat_b = matrix_to_quaternion(pose_b[..., :3, :3])

    dot = torch.sum(quat_a * quat_b, dim=-1, keepdim=True)
    quat_b = torch.where(dot < 0, -quat_b, quat_b)

    quaternion = quaternion_slerp(quat_a, quat_b, fraction)
    quaternion = torch.nn.functional.normalize(quaternion + torch.randn_like(quaternion) * noise_strengths[0], dim=-1)

    R = quaternion_to_matrix(quaternion)
    T = (1 - fraction) * pose_a[..., :3, 3] + fraction * pose_b[..., :3, 3]
    T = T + torch.randn_like(T) * noise_strengths[1]

    new_pose = pose_a.clone()
    new_pose[..., :3, :3] = R
    new_pose[..., :3, 3] = T
    return new_pose

def sample_from_dense_cameras(dense_cameras, t, noise_strengths=[0.01, 0.01]):
    _, N, A, B = dense_cameras.shape
    _, M = t.shape

    t = t.to(dense_cameras.device)
    left = torch.floor(t * (N - 1)).long().clamp(0, N - 2)
    right = left + 1
    fraction = (t * (N - 1) - left).unsqueeze(-1)
    a = torch.gather(dense_cameras, 1, repeat(left, 'a b -> a b A B', A=A, B=B))
    b = torch.gather(dense_cameras, 1, repeat(right, 'a b -> a b A B', A=A, B=B))

    new_pose = sample_from_two_pose(a[:, :, :3, 3:], b[:, :, :3, 3:], fraction, noise_strengths=noise_strengths)

    new_ins = (1 - fraction.unsqueeze(-1)) * a[:, :, :3, :3] + fraction.unsqueeze(-1) * b[:, :, :3, :3]

    return torch.cat([new_ins, new_pose], dim=-1)