import logging; log = logging.getLogger(__name__)
from .base import TextureFormat, fmts, types
from . import rgb, BCn
from . import formatinfo


def _register(cls):
    for sub in cls.__subclasses__():
        if sub.id is not None:
            fmts[sub.id] = sub
        _register(sub)

_register(TextureFormat)
