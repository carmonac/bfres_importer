"""Tests for the uking_mat combiner decoding (no Blender needed)."""
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
    """Horse_Link Mt_Body without its unsupported stage (R2, const_color12):
    coat = mix(cc0, cc1, Alb.R); socks = mix(coat, cc2, 1 - Sox.A);
    mouth over it by its alpha; times Spm alpha."""
    o = {}
    stage(o, 0, 4, [(100, 0), (101, 0), (0, 10), (400, 0)])
    stage(o, 1, 4, [(200, 0), (102, 0), (3, 41), (400, 0)])
    stage(o, 2, 4, [(201, 0), (103, 0), (300, 10), (400, 0)])
    stage(o, 3, 4, [(202, 0), (4, 0), (4, 40), (0, 0)])
    stage(o, 4, 2, [(203, 0), (1, 40), (0, 0), (0, 0)])
    o['uking_albedo_color'] = '204'
    return o


class CombinerTest(unittest.TestCase):
    def comb(self, opts, slots=5, vc=True):
        return C.Combiner(opts, slots, vc, range(6), range(8))

    def test_texture_and_channel(self):
        c = self.comb({'uking_specmask_scaler': '1', 'uking_specular_channel': '30'})
        e = c.parse_option('uking_specmask_scaler', 'uking_specular_channel')
        self.assertIsInstance(e, C.Channel)
        self.assertEqual((e.comp, e.invert), (3, False))
        self.assertEqual(e.expr.slot, 1)

    def test_one_minus(self):
        e = self.comb({}).parse(300, 41)
        self.assertEqual((e.comp, e.invert), (4, True))
        self.assertIsInstance(e.expr, C.VertexColor)

    def test_unsupported(self):
        c = self.comb({})
        for code, ch in ((503, 0), (0, 2), (0, 33), (7, 0), (106, 0), (408, 0)):
            with self.assertRaises(C.Unsupported):
                c.parse(code, ch)
        with self.assertRaises(C.Unsupported):
            self.comb({}, vc=False).parse(300, 10)

    def test_disabled_and_unknown_stage(self):
        o = {}
        stage(o, 0, 12, [(0, 0), (1, 0), (2, 0), (3, 0)])
        c = self.comb(o)
        with self.assertRaises(C.Unsupported):
            c.parse(200)      # calc type 12
        with self.assertRaises(C.Unsupported):
            c.parse(201)      # not enabled
        self.assertIsInstance(c.try_parse('uking_albedo_color'), C.Tex)  # default 0: slot 0
        o['uking_albedo_color'] = '200'
        self.assertIsNone(c.try_parse('uking_albedo_color'))

    def test_horse_chain(self):
        c = self.comb(horse_opts())
        e = c.parse_option('uking_albedo_color')
        self.assertEqual(C.textures_used(e), {0, 1, 3, 4})
        brown, dark, sock, sock2 = (0.18, 0.16, 0.15, 1), (0.02, 0.01, 0.01, 1), \
            (0.9, 0.9, 0.9, 1), (0.5, 0.5, 0.5, 1)
        mouth = (1.0, 0.0, 0.0, 0.0)  # transparent: no mouth here
        spm = (0, 0, 0, 0.5)
        args = dict(vertex_color=(0, 0, 0, 1),
                    const_colors={0: brown, 1: dark, 2: sock, 3: sock2},
                    const_values={})
        # no pattern (Alb.R 0), no socks (Sox.A 1): brown coat x Spm alpha
        tex = {0: (0, 0, 0, 1), 1: spm, 3: (1, 1, 1, 1), 4: mouth}
        r = C.evaluate(e, tex, **args)
        for x, y in zip(r[:3], brown[:3]):
            self.assertAlmostEqual(x, y * 0.5)
        # socks (Sox.A 0): sock color
        tex[3] = (0, 0, 0, 0)
        r = C.evaluate(e, tex, **args)
        self.assertAlmostEqual(r[0], 0.9 * 0.5)
        # mouth fully opaque wins
        tex[4] = (1.0, 0.0, 0.0, 1.0)
        r = C.evaluate(e, tex, **args)
        self.assertEqual(r[:3], (0.5, 0.0, 0.0))

    def test_clamp_and_product(self):
        o = {}
        stage(o, 0, 2, [(400, 0), (401, 0)], clamp=True)
        c = C.Combiner(o, 0, False, [], [0, 1])
        e = c.parse(200)
        self.assertEqual(C.evaluate(e, {}, None, {}, {0: 3.0, 1: 2.0}), (1.0,) * 4)
        self.assertEqual(C.evaluate(c.parse(200, 40), {}, None, {}, {0: 0.5, 1: 0.5}), (0.25,) * 4)

    def test_stage_cache(self):
        c = self.comb(horse_opts())
        self.assertIs(c.stage(0), c.stage(0))


if __name__ == '__main__':
    unittest.main()
