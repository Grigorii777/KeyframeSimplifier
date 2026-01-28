from dataclasses import dataclass
import enum
from typing import Any
import numpy as np

from cvat_tool.keyframes import distance_l1, distance_l2, distance_max


Vector = np.ndarray[Any, np.dtype[np.float_]]

class KeyframeSimplifyingMethod(enum.Enum):
    L_INF = "l_inf"
    L1 = "l1"
    L2 = "l2"

DISTANCE_FUNCTIONS = {
    KeyframeSimplifyingMethod.L_INF: distance_max,
    KeyframeSimplifyingMethod.L1: distance_l1,
    KeyframeSimplifyingMethod.L2: distance_l2,
}


@dataclass
class Keyframe:
    frame_id: int
    position: Vector
    rotation: Vector
    scale: Vector

class KeyframeField(enum.Enum):
    POSITION = "position"
    ROTATION = "rotation"
    SCALE = "scale"

@dataclass
class KeyframesField:
    keyframe_field: KeyframeField
    threshold: float
    method: KeyframeSimplifyingMethod
