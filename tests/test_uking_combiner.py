"""Tests for the uking_mat combiner decoding (no Blender needed).

The expected values come from the game's compiled shaders (see
bfres/Importer/uking_combiner.py and tools/uking_shader.py)."""
import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "uking_combiner", os.path.join(ROOT, "bfres", "Importer", "uking_combiner.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def stage(opts, n, calc, args, clamp=False):
    opts['uking_enable_calc_color%d' % n] = '1'
    opts['uking_color%d_calc_type' % n] = str(calc)
    opts['uking_color%d_clamp01' % n] = '1' if clamp else '0'
    for a, (src, ch) in zip('ABCD', args):
        opts['uking_color%d_%s' % (n, a)] = str(src)
        opts['uking_color%d_%s_channel' % (n, a)] = str(ch)


def horse_opts():
    """Horse_Link Mt_Body: coat = mix(cc0, cc1, Alb.R); socks; coat color
    cc4 (112) where Sox.R; cc3 by vertex red; mouth over it by its alpha;
    times Spm alpha. Slots: 0 _a0 Alb, 1 _s0 Spm, 3 _e0 Sox, 4 _t0 Mouth."""
    o = {}
    stage(o, 0, 4, [(100, 0), (101, 0), (0, 10), (400, 0)])
    stage(o, 1, 4, [(200, 0), (102, 0), (3, 41), (400, 0)])
    stage(o, 2, 4, [(201, 0), (112, 0), (3, 10), (400, 0)])
    stage(o, 3, 4, [(202, 0), (103, 0), (300, 10), (400, 0)])
    stage(o, 4, 4, [(203, 0), (4, 0), (4, 40), (0, 0)])
    stage(o, 5, 2, [(204, 0), (1, 40), (0, 0), (0, 0)])
    o['uking_albedo_color'] = '205'
    return o


def moss_opts():
    """Mt_Rock_PlayerChapelCutsurface_A (FldObj_PlayerChapel_A): the IR
    computes m = clamp(2a) * mix(MossAlb.a, 1, clamp(2a - 1)), a = vertex
    alpha, and blends rock/moss albedo and normal blue by m."""
    o = {}
    stage(o, 0, 5, [(300, 40), (401, 2), (403, 0), (400, 0)], clamp=True)
    stage(o, 1, 5, [(300, 40), (400, 0), (403, 0), (400, 0)], clamp=True)
    stage(o, 2, 12, [(402, 40), (2, 40), (200, 41), (201, 40)])
    stage(o, 3, 4, [(0, 0), (2, 0), (202, 40), (400, 0)])
    o['uking_albedo_color'] = '203'
    return o


class CombinerTest(unittest.TestCase):
    def comb(self, opts, slots=6, vc=True):
        return C.Combiner(opts, slots, vc, range(6), range(8))

    def test_sources(self):
        c = self.comb({})
        self.assertEqual(c.parse(3).slot, 3)
        self.assertEqual(c.parse(102).index, 2)                 # const_color2
        self.assertIsInstance(c.parse(102), C.ConstColor)
        self.assertIsInstance(c.parse(104), C.ConstValue)        # const_value0
        self.assertEqual(c.parse(111).index, 7)
        self.assertEqual(c.parse(112).index, 4)                 # const_color4
        self.assertIsInstance(c.parse(113), C.ConstColor)
        self.assertEqual([c.parse(v).value for v in (400, 401, 402, 403)], [0, 0.5, 1, 2])
        self.assertEqual(c.parse(503).code, C.RIM_NORMAL)
        self.assertIsInstance(c.parse(300), C.VertexColor)

    def test_texture_and_channel(self):
        c = self.comb({'uking_specmask_scaler': '1', 'uking_specular_channel': '30'})
        e = c.parse_option('uking_specmask_scaler', 'uking_specular_channel')
        self.assertIsInstance(e, C.Channel)
        self.assertEqual((e.comp, e.mod), (3, 0))
        self.assertEqual(e.expr.slot, 1)

    def test_channel_modifiers(self):
        e = self.comb({}).parse(300, 41)
        self.assertEqual((e.comp, e.mod), (4, 1))
        e = self.comb({}).parse(105, 2)
        self.assertEqual((e.comp, e.mod), (0, 2))
        r = C.evaluate(e, {}, None, {}, {1: 0.4})
        self.assertEqual(r, (-0.4,) * 4)

    def test_unsupported(self):
        c = self.comb({}, slots=3)
        for code, ch in ((500, 0), (8, 0), (0, 3), (0, 1001), (5, 0), (114, 0), (404, 0)):
            with self.assertRaises(C.Unsupported):
                c.parse(code, ch)
        with self.assertRaises(C.Unsupported):
            self.comb({}, vc=False).parse(300, 10)

    def test_disabled_and_unknown_stage(self):
        o = {}
        stage(o, 0, 3, [(0, 0), (1, 0), (2, 0), (3, 0)])
        c = self.comb(o)
        with self.assertRaises(C.Unsupported):
            c.parse(200)      # calc type 3
        with self.assertRaises(C.Unsupported):
            c.parse(201)      # not enabled
        self.assertIsInstance(c.try_parse('uking_albedo_color'), C.Tex)  # default 0: slot 0
        o['uking_albedo_color'] = '200'
        self.assertIsNone(c.try_parse('uking_albedo_color'))

    def test_horse_chain(self):
        c = self.comb(horse_opts())
        e = c.parse_option('uking_albedo_color')
        self.assertEqual(C.textures_used(e), {0, 1, 3, 4})
        brown, dark, sock = (0.18, 0.16, 0.15, 1), (0.02, 0.01, 0.01, 1), (0.9, 0.9, 0.9, 1)
        coat2, tint = (0.5, 0.4, 0.3, 1), (1, 1, 1, 1)
        args = dict(vertex_color=(0, 0, 0, 1), const_colors={0: brown, 1: dark, 2: sock,
                    3: tint, 4: coat2}, const_values={})
        spm = (0, 0, 0, 0.5)
        mouth = (1.0, 0.0, 0.0, 0.0)  # transparent: no mouth here
        # no pattern, no socks (Sox all 1 -> 1-A = 0, R = 1 -> coat2)
        tex = {0: (0, 0, 0, 1), 1: spm, 3: (0, 0, 0, 1), 4: mouth}
        r = C.evaluate(e, tex, **args)
        for x, y in zip(r[:3], brown[:3]):
            self.assertAlmostEqual(x, y * 0.5)
        tex[3] = (0, 0, 0, 0)          # socks
        self.assertAlmostEqual(C.evaluate(e, tex, **args)[0], 0.9 * 0.5)
        tex[4] = (1.0, 0.0, 0.0, 1.0)  # mouth fully opaque wins
        self.assertEqual(C.evaluate(e, tex, **args)[:3], (0.5, 0.0, 0.0))

    def test_moss_mask(self):
        c = self.comb(moss_opts())
        e = c.parse_option('uking_albedo_color')
        rock, moss = (0.2, 0.2, 0.2, 1), (0.1, 0.6, 0.1, 0.25)
        for a in (0.0, 0.2, 0.5, 0.7, 1.0):
            m = min(max(2 * a, 0), 1) * ((1 - min(max(2 * a - 1, 0), 1)) * moss[3]
                                         + min(max(2 * a - 1, 0), 1))
            r = C.evaluate(e, {0: rock, 2: moss}, (0, 0, 0, a), {}, {})
            self.assertAlmostEqual(r[1], rock[1] * (1 - m) + moss[1] * m)

    def test_formulas(self):
        """Each calc type against its formula with distinct values."""
        A, B, Cc, D = 0.3, 0.6, 0.25, 0.8
        want = {1: A + B, 2: A * B, 4: A * (1 - Cc) + B * Cc, 5: (A + B) * Cc,
                6: Cc * D - A * B, 7: A ** B, 8: A * B + Cc, 9: A * B * Cc,
                11: A * B * Cc * D, 12: (A * (1 - Cc) + B * Cc) * D,
                15: A * (1 - Cc) + B * Cc + D, 21: (A * B + Cc) * D, 28: A ** B * Cc}
        for t, v in want.items():
            o = {}
            stage(o, 0, t, [(104, 0), (105, 0), (106, 0), (107, 0)])
            c = C.Combiner(o, 0, False, [], range(8))
            r = C.evaluate(c.parse(200), {}, None, {}, {0: A, 1: B, 2: Cc, 3: D})
            self.assertAlmostEqual(r[0], v, msg='calc type %d' % t)

    def test_clamp(self):
        o = {}
        stage(o, 0, 2, [(403, 0), (403, 0)], clamp=True)
        c = C.Combiner(o, 0, False, [], [])
        self.assertEqual(C.evaluate(c.parse(200), {}, None, {}, {}), (1.0,) * 4)

    def test_stage_cache(self):
        c = self.comb(horse_opts())
        self.assertIs(c.stage(0), c.stage(0))


if __name__ == '__main__':
    unittest.main()
