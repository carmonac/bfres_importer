"""Vectorized transform math for animation import (numpy only).

Matrices are (n, 4, 4), column vectors (M @ p), like mathutils.
Quaternions are (n, 4) in Blender order (w, x, y, z).
"""
import numpy as np


def euler_xyz_to_matrix(e):
    """Euler XYZ (radians, (n, 3)) -> rotation matrices (n, 3, 3).

    Same convention as mathutils.Euler(order='XYZ') and BFRES: rotate about
    X, then Y, then Z, i.e. R = Rz @ Ry @ Rx.
    """
    e = np.asarray(e, dtype=np.float64)
    cx, cy, cz = np.cos(e[:, 0]), np.cos(e[:, 1]), np.cos(e[:, 2])
    sx, sy, sz = np.sin(e[:, 0]), np.sin(e[:, 1]), np.sin(e[:, 2])
    R = np.empty((len(e), 3, 3))
    R[:, 0, 0] = cy * cz
    R[:, 0, 1] = sx * sy * cz - cx * sz
    R[:, 0, 2] = cx * sy * cz + sx * sz
    R[:, 1, 0] = cy * sz
    R[:, 1, 1] = sx * sy * sz + cx * cz
    R[:, 1, 2] = cx * sy * sz - sx * cz
    R[:, 2, 0] = -sy
    R[:, 2, 1] = sx * cy
    R[:, 2, 2] = cx * cy
    return R


def quat_to_matrix(q):
    """Quaternions (n, 4) as (w, x, y, z) -> rotation matrices (n, 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def matrix_to_quat(R):
    """Rotation matrices (n, 3, 3) -> unit quaternions (n, 4), (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64)
    n = len(R)
    q = np.empty((n, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    c0 = tr > 0
    c1 = ~c0 & (R[:, 0, 0] >= R[:, 1, 1]) & (R[:, 0, 0] >= R[:, 2, 2])
    c2 = ~c0 & ~c1 & (R[:, 1, 1] >= R[:, 2, 2])
    c3 = ~c0 & ~c1 & ~c2
    if c0.any():
        s = np.sqrt(tr[c0] + 1.0) * 2
        r = R[c0]
        q[c0] = np.stack((0.25 * s, (r[:, 2, 1] - r[:, 1, 2]) / s,
                          (r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 1, 0] - r[:, 0, 1]) / s), 1)
    if c1.any():
        r = R[c1]
        s = np.sqrt(1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2]) * 2
        q[c1] = np.stack(((r[:, 2, 1] - r[:, 1, 2]) / s, 0.25 * s,
                          (r[:, 0, 1] + r[:, 1, 0]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s), 1)
    if c2.any():
        r = R[c2]
        s = np.sqrt(1.0 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2]) * 2
        q[c2] = np.stack(((r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 0, 1] + r[:, 1, 0]) / s,
                          0.25 * s, (r[:, 1, 2] + r[:, 2, 1]) / s), 1)
    if c3.any():
        r = R[c3]
        s = np.sqrt(1.0 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1]) * 2
        q[c3] = np.stack(((r[:, 1, 0] - r[:, 0, 1]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s,
                          (r[:, 1, 2] + r[:, 2, 1]) / s, 0.25 * s), 1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def compose(t, R, s):
    """Translation (n,3), rotation matrices (n,3,3), scale (n,3) -> (n,4,4)
    = T @ R @ S, like mathutils.Matrix.LocRotScale."""
    n = len(t)
    M = np.zeros((n, 4, 4))
    M[:, :3, :3] = R * np.asarray(s, dtype=np.float64)[:, None, :]
    M[:, :3, 3] = t
    M[:, 3, 3] = 1.0
    return M


def decompose(M):
    """(n,4,4) -> translation (n,3), quaternion (n,4), scale (n,3).

    Negative scale is not recovered (it is not used by bones in practice).
    """
    t = M[:, :3, 3].copy()
    A = M[:, :3, :3]
    s = np.linalg.norm(A, axis=1)
    s_safe = np.where(s == 0, 1.0, s)
    R = A / s_safe[:, None, :]
    return t, matrix_to_quat(R), s


def make_continuous(q):
    """Flip quaternion signs so consecutive frames take the short way."""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


ROT_X_90 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def bone_local(scale, rotate, translate, euler, is_root):
    """Parent-relative bone matrices (n,4,4) from sampled game values, in
    the convention SkeletonImporter uses for the rest pose: LocRotScale of
    the game values, with root bones also rotated from Y-up to Z-up."""
    if euler:
        R = euler_xyz_to_matrix(rotate[:, :3])
    else:
        # game quaternions are (x, y, z, w)
        R = quat_to_matrix(np.stack((rotate[:, 3], rotate[:, 0], rotate[:, 1], rotate[:, 2]), 1))
    M = compose(translate, R, scale)
    if is_root:
        M = ROT_X_90 @ M
    return M


def pose_basis(rest_local, anim_local):
    """Blender pose-bone basis matrices: rest_local^-1 @ anim_local."""
    return np.linalg.inv(rest_local) @ anim_local
