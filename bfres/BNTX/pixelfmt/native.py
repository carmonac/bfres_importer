"""Optional native texture decoder: texture2ddecoder (MIT, K0lb3/Perfare).

The pure-Python (numpy) decoders in this package handle every format but
ASTC; the native module is much faster and adds ASTC. It is found, in order:

1. ``import texture2ddecoder``: installed in Blender's Python, or installed
   by Blender from the wheels listed in ``blender_manifest.toml`` when the
   add-on is installed as an extension (Blender 4.2+).
2. A wheel in the add-on's ``wheels/`` folder matching this platform and
   Python. This covers loading the add-on straight from a checkout (the
   test harness, ``--factory-startup``): a wheel is a zip file and native
   code cannot be imported from inside a zip, so the wheel is extracted
   once into a per-user cache folder (``BFRES_CACHE_DIR``, else
   ``~/Library/Caches/bfres_importer`` on macOS,
   ``%LOCALAPPDATA%\\bfres_importer`` on Windows,
   ``$XDG_CACHE_HOME/bfres_importer`` or ``~/.cache/bfres_importer``
   elsewhere) and imported from there. The folder is named after the
   wheel, so a new version gets a new folder.

``BFRES_NATIVE_DECODER=0`` in the environment disables the native module,
e.g. to test or time the Python decoders; ``set_enabled()`` does the same
at runtime.
"""
import logging; log = logging.getLogger(__name__)
import importlib.util
import os
import platform
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
WHEELS = os.path.join(ROOT, 'wheels')
PACKAGE = 'texture2ddecoder'
MODNAME = '_bfres_texture2ddecoder'   # private name: never clashes with
                                      # a texture2ddecoder of another add-on

_module = None
_tried = False
_enabled = os.environ.get('BFRES_NATIVE_DECODER', '1').lower() not in ('0', 'no', 'false', 'off')
source = None   # where the module came from, for logs and the test report


def set_enabled(on):
    """Turn the native decoder on or off (the Python decoders are used
    while it is off)."""
    global _enabled
    _enabled = bool(on)


def get():
    """The texture2ddecoder module, or None."""
    global _module, _tried, source
    if not _enabled:
        return None
    if not _tried:
        _tried = True
        try:
            _module, source = _load()
        except Exception as ex:
            log.warning("Native texture decoder unavailable (%s): using "
                "the Python decoders (no ASTC support)", ex)
            _module, source = None, None
        if _module is not None:
            log.info("Native texture decoder: %s", source)
    return _module


def _load():
    try:
        import texture2ddecoder
        return texture2ddecoder, texture2ddecoder.__file__
    except ImportError:
        pass
    wheel = find_wheel()
    if wheel is None:
        raise ImportError("no texture2ddecoder installed and no wheel for "
            "%s / Python %d.%d in %s" % (_platform_desc(),
            sys.version_info[0], sys.version_info[1], WHEELS))
    path = _extract(wheel)
    init = os.path.join(path, PACKAGE, '__init__.py')
    spec = importlib.util.spec_from_file_location(MODNAME, init,
        submodule_search_locations=[os.path.join(path, PACKAGE)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[MODNAME] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[MODNAME]
        raise
    return mod, os.path.basename(wheel)


# ------------------------------------------------------------ wheel lookup

def _platform_desc():
    return "%s %s" % (sys.platform, platform.machine())


def _platform_ok(tag):
    """Does a wheel platform tag (e.g. macosx_11_0_arm64) fit this system?"""
    machine = platform.machine().lower()
    arch = {'amd64': 'x86_64', 'x64': 'x86_64', 'arm64': 'arm64',
            'aarch64': 'aarch64'}.get(machine, machine)
    for t in tag.split('.'):
        if sys.platform == 'darwin' and t.startswith('macosx_'):
            if t.endswith('_universal2') or t.endswith('_' + arch):
                return True
        elif sys.platform == 'win32':
            if t == {'x86_64': 'win_amd64', 'arm64': 'win_arm64'}.get(arch):
                return True
        elif sys.platform.startswith('linux'):
            if (t.startswith('manylinux') or t.startswith('linux')) and t.endswith('_' + arch):
                return True
    return False


def _python_ok(pytag, abitag):
    """cp313-cp313 needs exactly 3.13; cp311-abi3 works on 3.11 and later."""
    cur = (sys.version_info[0], sys.version_info[1])
    for p in pytag.split('.'):
        if not p.startswith('cp') or not p[2:].isdigit():
            continue
        v = (int(p[2]), int(p[3:]))
        if abitag == 'abi3' and v <= cur:
            return True
        if v == cur:
            return True
    return False


def find_wheel(folder=WHEELS):
    """Best matching texture2ddecoder wheel in `folder`, or None."""
    if not os.path.isdir(folder):
        return None
    best = None
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith('.whl'):
            continue
        parts = fn[:-4].split('-')
        if len(parts) < 5 or parts[0] != PACKAGE:
            continue
        pytag, abitag, plat = parts[-3:]
        if _python_ok(pytag, abitag) and _platform_ok(plat):
            best = os.path.join(folder, fn)   # sorted: highest version wins
    return best


def cache_dir():
    d = os.environ.get('BFRES_CACHE_DIR')
    if d:
        return d
    home = os.path.expanduser('~')
    if sys.platform == 'darwin':
        return os.path.join(home, 'Library', 'Caches', 'bfres_importer')
    if sys.platform == 'win32':
        base = os.environ.get('LOCALAPPDATA') or os.path.join(home, 'AppData', 'Local')
        return os.path.join(base, 'bfres_importer')
    base = os.environ.get('XDG_CACHE_HOME') or os.path.join(home, '.cache')
    return os.path.join(base, 'bfres_importer')


def _extract(wheel):
    """Unzip `wheel` into the cache once; returns the folder."""
    name = os.path.basename(wheel)[:-4]
    root = os.path.join(cache_dir(), 'wheels')
    dest = os.path.join(root, name)
    marker = os.path.join(dest, '.complete')
    if os.path.exists(marker):
        return dest
    os.makedirs(root, exist_ok=True)
    tmp = '%s.tmp%d' % (dest, os.getpid())
    with zipfile.ZipFile(wheel) as z:
        for info in z.infolist():
            # refuse absolute paths and '..' (never in a real wheel)
            parts = info.filename.replace('\\', '/').split('/')
            if info.filename.startswith('/') or '..' in parts:
                raise ValueError("unsafe path %r in %s" % (info.filename, wheel))
        z.extractall(tmp)
    open(os.path.join(tmp, '.complete'), 'w').close()
    try:
        os.replace(tmp, dest)   # atomic; fails if another process won
    except OSError:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        if not os.path.exists(marker):
            raise
    return dest


# ------------------------------------------------------------ decoding

def _bgra(out, width, height):
    px = np.frombuffer(out, np.uint8)
    if px.size != width * height * 4:
        raise ValueError("native decoder returned %d bytes for %dx%d"
            % (px.size, width, height))
    return px.reshape(height, width, 4)


def rgba(func, data, width, height, *args):
    """Run decoder `func` (e.g. 'decode_bc7') -> uint8 (height, width, 4)
    RGBA, or None without the native module."""
    mod = get()
    if mod is None:
        return None
    px = _bgra(getattr(mod, func)(bytes(data), width, height, *args), width, height)
    return px[..., [2, 1, 0, 3]]


def channels(func, data, width, height, which):
    """Like rgba() but only the RGBA channels in `which`, e.g. (0, 1)."""
    mod = get()
    if mod is None:
        return None
    px = _bgra(getattr(mod, func)(bytes(data), width, height), width, height)
    bgra = {0: 2, 1: 1, 2: 0, 3: 3}
    return px[..., [bgra[c] for c in which]]
