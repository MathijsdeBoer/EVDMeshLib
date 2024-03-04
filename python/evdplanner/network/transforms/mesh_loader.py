from pathlib import Path
from typing import Hashable, Mapping

import monai.transforms as mt

from evdplanner.geometry import Mesh


class MeshLoaderd(mt.MapTransform):
    def __init__(self, keys: list[str], allow_missing_keys: bool = False) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, Path]) -> dict:
        d = dict(data)

        for key in self.key_iterator(d):
            d[key] = Mesh.load(str(d[key]))

        return d
