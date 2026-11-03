"""TerrainTextures file lookup (no Blender needed)."""
import importlib.util
import os
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "terrain_textures", os.path.join(ROOT, "bfres", "Importer", "TerrainTextures.py"))
try:
    TT = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TT)
except ImportError:  # numpy missing
    TT = None


@unittest.skipIf(TT is None, "numpy not available")
class FindTerrainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.models = os.path.join(self.root, 'botw')
        os.makedirs(self.models)
        self.model = os.path.join(self.models, 'FldObj_Mound_A.sbfres')
        open(self.model, 'wb').close()
        os.environ.pop('BFRES_TERRAIN_TEX', None)

    def tearDown(self):
        self.tmp.cleanup()

    def touch(self, *parts):
        p = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'wb').close()
        return p

    def test_none(self):
        self.assertIsNone(TT.find_terrain_file(self.model))

    def test_next_to_model(self):
        p = self.touch('botw', 'Terrain.Tex.sbfres')
        self.assertEqual(TT.find_terrain_file(self.model), p)

    def test_parent_and_model_folder(self):
        p = self.touch('Model', 'Terrain.Tex.sbfres')
        self.assertEqual(TT.find_terrain_file(self.model), p)
        q = self.touch('Terrain.Tex.bfres')
        self.assertEqual(TT.find_terrain_file(self.model), q)  # parent first

    def test_explicit_and_env(self):
        p = self.touch('elsewhere', 'T.sbfres')
        self.assertEqual(TT.find_terrain_file(self.model, p), p)
        os.environ['BFRES_TERRAIN_TEX'] = p
        try:
            self.assertEqual(TT.find_terrain_file(self.model), p)
        finally:
            del os.environ['BFRES_TERRAIN_TEX']

    def test_uncompressed_passthrough(self):
        p = self.touch('botw', 'Terrain.Tex.bfres')
        with open(p, 'wb') as f:
            f.write(b'FRES    ')
        self.assertEqual(TT._decompressed(p), p)

    def test_image_name(self):
        self.assertEqual(TT.image_name('MaterialAlb', 18), 'MaterialAlb[18]')


if __name__ == '__main__':
    unittest.main()
