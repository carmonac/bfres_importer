"""Animation transform math tests. Needs numpy."""
import importlib.util
import math
import os
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    spec = importlib.util.spec_from_file_location(
        "animmath", os.path.join(ROOT, "bfres", "Importer", "animmath.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipIf(np is None, "numpy not available")
class AnimMathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load()

    def test_euler_axes(self):
        R = self.m.euler_xyz_to_matrix(np.array([[math.pi / 2, 0, 0]]))[0]
        # +Y rotated 90 degrees about X goes to +Z
        np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, 1], atol=1e-12)
        R = self.m.euler_xyz_to_matrix(np.array([[0, 0, math.pi / 2]]))[0]
        np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_euler_order_is_x_then_y_then_z(self):
        e = np.array([[0.3, -0.7, 1.1]])
        R = self.m.euler_xyz_to_matrix(e)[0]
        rx = self.m.euler_xyz_to_matrix(np.array([[0.3, 0, 0]]))[0]
        ry = self.m.euler_xyz_to_matrix(np.array([[0, -0.7, 0]]))[0]
        rz = self.m.euler_xyz_to_matrix(np.array([[0, 0, 1.1]]))[0]
        np.testing.assert_allclose(R, rz @ ry @ rx, atol=1e-12)

    def test_quat_roundtrip(self):
        rng = np.random.default_rng(3)
        q = rng.normal(size=(200, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        q2 = self.m.matrix_to_quat(self.m.quat_to_matrix(q))
        # same rotation up to sign
        self.assertTrue(np.allclose(np.abs(np.sum(q * q2, axis=1)), 1, atol=1e-9))

    def test_compose_decompose(self):
        rng = np.random.default_rng(4)
        e = rng.uniform(-3, 3, (50, 3))
        t = rng.normal(size=(50, 3))
        s = rng.uniform(0.2, 3, (50, 3))
        M = self.m.compose(t, self.m.euler_xyz_to_matrix(e), s)
        t2, q, s2 = self.m.decompose(M)
        np.testing.assert_allclose(t2, t)
        np.testing.assert_allclose(s2, s)
        M2 = self.m.compose(t2, self.m.quat_to_matrix(q), s2)
        np.testing.assert_allclose(M2, M, atol=1e-9)

    def test_rest_pose_gives_identity_basis(self):
        rest = self.m.bone_local(np.ones((1, 3)), np.array([[0.4, 0.1, -0.2, 1]]),
                                 np.array([[1.0, 2.0, 3.0]]), True, True)
        anim = self.m.bone_local(np.ones((3, 3)), np.tile([0.4, 0.1, -0.2, 1], (3, 1)),
                                 np.tile([1.0, 2.0, 3.0], (3, 1)), True, True)
        basis = self.m.pose_basis(rest[0], anim)
        np.testing.assert_allclose(basis, np.tile(np.eye(4), (3, 1, 1)), atol=1e-12)

    def test_game_quaternion_order(self):
        # game stores (x, y, z, w); 90 degrees about Z
        h = math.sqrt(0.5)
        M = self.m.bone_local(np.ones((1, 3)), np.array([[0, 0, h, h]]),
                              np.zeros((1, 3)), False, False)[0]
        np.testing.assert_allclose(M[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_srt_to_mapping_matches_flipped_transform(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            sx, sy = rng.uniform(0.2, 3, 2)
            rot = rng.uniform(-3, 3)
            tx, ty = rng.normal(size=2)
            (lx, ly), r, (ssx, ssy) = self.m.srt_to_mapping(sx, sy, rot, tx, ty)
            for u, v in rng.uniform(-1, 2, (5, 2)):
                # game: uv' = T + R(rot) S uv, evaluated in game space
                c, s_ = math.cos(rot), math.sin(rot)
                vg = 1 - v
                gu = tx + c * sx * u - s_ * sy * vg
                gv = ty + s_ * sx * u + c * sy * vg
                # blender mapping node: loc + R(r) (scale * uv)
                cb, sb = math.cos(r), math.sin(r)
                bu = lx + cb * ssx * u - sb * ssy * v
                bv = ly + sb * ssx * u + cb * ssy * v
                self.assertAlmostEqual(bu, gu, places=9)
                self.assertAlmostEqual(bv, 1 - gv, places=9)

    def test_srt_identity(self):
        (lx, ly), r, (sx, sy) = self.m.srt_to_mapping(1, 1, 0, 0, 0)
        self.assertEqual((float(lx), float(ly), float(r), float(sx), float(sy)), (0, 0, 0, 1, 1))

    def test_continuity(self):
        q = np.array([[1, 0, 0, 0], [-0.99, -0.1, 0, 0], [0.98, 0.2, 0, 0]], float)
        c = self.m.make_continuous(q)
        self.assertTrue((np.sum(c[1:] * c[:-1], axis=1) > 0).all())

    def _rand_rigid(self, rng, n, scale=None):
        R = self.m.quat_to_matrix(rng.normal(size=(n, 4)))
        t = rng.normal(size=(n, 3))
        s = np.ones((n, 3)) if scale is None else scale
        return self.m.compose(t, R, s)

    def _chain(self, local):
        """World matrices of a chain of bones from parent-relative ones."""
        out, W = [], np.eye(4)
        for L in local:
            W = W @ L
            out.append(W)
        return out

    def test_offset_identity_is_plain_basis(self):
        rng = np.random.default_rng(5)
        rest = self._rand_rigid(rng, 1)[0]
        anim = self._rand_rigid(rng, 4)
        np.testing.assert_allclose(self.m.pose_basis(rest, anim, np.eye(4)),
                                   self.m.pose_basis(rest, anim), atol=1e-12)

    def _check_offset_chain(self, anim_scale):
        """A chain of bones whose Blender rest pose B = G @ O is reoriented
        from the game rest G: posing with the compensated basis puts every
        bone at A @ O (A = game animated matrix), so heads follow the game
        and skinning matrices pose @ B^-1 equal A @ G^-1."""
        rng = np.random.default_rng(6)
        nb, nf = 5, 7
        G = self._chain(self._rand_rigid(rng, nb))          # game rest, world
        O = []
        for _ in range(nb):                                # rotation only
            o = self._rand_rigid(rng, 1)[0]
            o[:3, 3] = 0
            O.append(o)
        B = [g @ o for g, o in zip(G, O)]                  # Blender rest
        anim = [self._rand_rigid(rng, nf, anim_scale(rng, nf)) for _ in range(nb)]
        for f in range(nf):
            A = self._chain([a[f] for a in anim])          # game animated
            P = np.eye(4)                                  # Blender pose world
            for b in range(nb):
                rest_game = G[b] if b == 0 else np.linalg.inv(G[b - 1]) @ G[b]
                offset = np.linalg.inv(G[b]) @ B[b]
                basis = self.m.pose_basis(rest_game, anim[b][f:f + 1], offset)[0]
                # Blender: pose = parent_pose @ (B_parent^-1 @ B) @ basis
                rest_blender = B[b] if b == 0 else np.linalg.inv(B[b - 1]) @ B[b]
                P = P @ rest_blender @ basis
                np.testing.assert_allclose(P, A[b] @ O[b], atol=1e-9)
                np.testing.assert_allclose(P[:3, 3], A[b][:3, 3], atol=1e-9)
                np.testing.assert_allclose(P @ np.linalg.inv(B[b]),
                                           A[b] @ np.linalg.inv(G[b]), atol=1e-9)

    def test_offset_compensation_rigid(self):
        self._check_offset_chain(lambda rng, n: None)

    def test_offset_compensation_uniform_scale(self):
        self._check_offset_chain(
            lambda rng, n: np.repeat(rng.uniform(0.5, 2, (n, 1)), 3, axis=1))

    def test_offset_basis_decomposes(self):
        # with rigid or uniformly scaled animation the compensated basis is
        # still a plain loc/rot/scale (no shear), so keyframes reproduce it
        rng = np.random.default_rng(7)
        rest = self._rand_rigid(rng, 1)[0]
        s = np.repeat(rng.uniform(0.5, 2, (6, 1)), 3, axis=1)
        anim = self._rand_rigid(rng, 6, s)
        o = self._rand_rigid(rng, 1)[0]
        o[:3, 3] = 0
        basis = self.m.pose_basis(rest, anim, o)
        t, q, sc = self.m.decompose(basis)
        np.testing.assert_allclose(self.m.compose(t, self.m.quat_to_matrix(q), sc),
                                   basis, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
