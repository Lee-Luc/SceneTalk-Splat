"""
2D到3D投影模块
实现检测框的跨视角投影
"""

from .bbox_projector import BBoxProjector
from .object_3d_reconstructor import Object3DReconstructor

__all__ = [
    'BBoxProjector',
    'Object3DReconstructor',
]

