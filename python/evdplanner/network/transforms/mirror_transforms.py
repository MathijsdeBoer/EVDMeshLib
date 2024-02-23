"""
Mirror Transform.
"""
import random
from typing import Sequence

import monai.transforms as mt
import torch


class MirrorTransform(mt.MapTransform, mt.InvertibleTransform):
    """
    Mirror the image and the label.
    """

    def __init__(
        self,
        keys: list[str],
        image_key: str = "image",
        label_key: str = "label",
        keypoint_pairs: Sequence[tuple[int, int]] = None,
        mirrorable_axes: Sequence[int] = None,
    ) -> None:
        """
        Initialize the transform.

        Parameters
        ----------
        keys : list[str]
            The keys to apply the transform to.
        image_key : str, optional
            The key for the image.
        label_key : str, optional
            The key for the label.
        """
        super().__init__(keys)
        self.image_key = image_key
        self.label_key = label_key
        self.keypoint_pairs = keypoint_pairs
        self.mirrorable_axes = mirrorable_axes

    def __call__(self, data: dict) -> dict:
        """
        Mirror the image and the label.

        Parameters
        ----------
        data : dict
            The input dictionary.

        Returns
        -------
        dict
            The output dictionary.
        """
        d = dict(data)

        axes_to_mirror = random.choice(self.mirrorable_axes)

        for key in self.key_iterator(d):
            if key == self.image_key:
                d[key] = torch.flip(d[key], [axes_to_mirror])

            if key == self.label_key:
                d[key][axes_to_mirror + 2] = 1 - d[key][axes_to_mirror + 2]

                for pair in self.keypoint_pairs:
                    tmp = d[key][:, :, pair[0]]
                    d[key][:, :, pair[0]] = d[key][:, :, pair[1]]
                    d[key][:, :, pair[1]] = tmp

        return d
