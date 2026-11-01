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


if __name__ == '__main__':
    unittest.main()
