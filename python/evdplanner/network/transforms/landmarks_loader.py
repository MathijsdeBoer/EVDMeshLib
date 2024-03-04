from pathlib import Path

import monai.transforms as mt

from evdplanner.linalg import Vec3
from evdplanner.markups import MarkupManager


class LandmarksLoaderd(mt.MapTransform):
    def __init__(self, keys: list[str], allow_missing_keys: bool = False) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: dict[str, Path]) -> dict:
        d = dict(data)

        for key in self.key_iterator(d):
            manager = MarkupManager.load(d[key])

            landmarks = []
            names = []
            for markup in manager.markups:
                for control_point in markup.control_points:
                    landmarks.append(Vec3(*control_point.position))
                    names.append(control_point.label)

            d[key] = landmarks, names
        return d
