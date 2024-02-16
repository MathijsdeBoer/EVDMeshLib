import json
from pathlib import Path
from typing import Hashable, Mapping

import monai.transforms as mt
import torch


class JsonKeypointLoaderd(mt.MapTransform):
    def __init__(
            self,
            json_key: str,
            output_key: str,
            keypoint_names: list[str] | None = None,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys=[json_key], allow_missing_keys=allow_missing_keys)
        self.json_key = json_key
        self.output_key = output_key
        self.keypoint_names = keypoint_names

    def __call__(self, data: Mapping[Hashable, Path]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)

        for key in self.key_iterator(d):
            with d[key].open("r") as f:
                keypoints = json.load(f)

            if self.keypoint_names:
                d[self.output_key] = torch.tensor(
                    [
                        x["position"]
                        for x in keypoints
                        if x["label"] in self.keypoint_names
                    ]
                )
            else:
                d[self.output_key] = torch.tensor([x["position"] for x in keypoints])

        return d
