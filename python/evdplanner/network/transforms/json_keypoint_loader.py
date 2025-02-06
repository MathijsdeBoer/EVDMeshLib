"""
A MONAI transform for loading keypoints from a JSON file.
"""

import json
from collections.abc import Hashable, Mapping
from pathlib import Path

import monai.transforms as mt
import torch


class JsonKeypointLoaderd(mt.MapTransform):
    """
    Load keypoints from a JSON file and store them in a tensor.
    """

    def __init__(
        self,
        json_key: str,
        output_key: str,
        keypoint_names: list[str] | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Initialize the transform.

        Parameters
        ----------
        json_key : str
            The key in the input dictionary that contains the path to the JSON file.
        output_key : str
            The key in the output dictionary to store the keypoints.
        keypoint_names : list[str], optional
            A list of keypoint names to extract from the JSON file. If None, all keypoints
            are extracted.
        allow_missing_keys : bool, optional
            If False, raise an exception if the input dictionary is missing the json_key.
            If True, do not raise an exception.
        """
        super().__init__(keys=[json_key], allow_missing_keys=allow_missing_keys)
        self.json_key = json_key
        self.output_key = output_key
        self.keypoint_names = keypoint_names

    def __call__(self, data: Mapping[Hashable, Path]) -> dict[Hashable, torch.Tensor]:
        """
        Load keypoints from a JSON file and store them in a tensor.

        Parameters
        ----------
        data : Mapping[Hashable, Path]
            The input dictionary.

        Returns
        -------
        dict[Hashable, torch.Tensor]
            The output dictionary.
        """
        d = dict(data)

        for key in self.key_iterator(d):
            with d[key].open("r") as f:
                keypoints = json.load(f)

            if self.keypoint_names:
                d[self.output_key] = torch.tensor(
                    [x["position"] for x in keypoints if x["label"] in self.keypoint_names],
                    dtype=torch.float32,
                )
            else:
                d[self.output_key] = torch.tensor([x["position"] for x in keypoints])

        return d
