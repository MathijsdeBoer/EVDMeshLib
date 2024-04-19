from collections.abc import Hashable, Mapping, Sequence

import monai.transforms as mt
import torch


def flip_keypoints(keypoints: torch.Tensor, pairs: tuple[tuple[int, int], ...]) -> Sequence[float]:
    """
    Flip keypoints horizontally and swap pairs of keypoints.

    Parameters
    ----------
    keypoints : torch.Tensor
        The keypoints to flip.
    pairs : tuple[tuple[int, int], ...]
        A list of pairs of indices to swap.

    Returns
    -------
    Sequence[float]
        The flipped keypoints.
    """
    keypoints = keypoints.clone()
    keypoints[:, 0] = 1 - keypoints[:, 0]
    for a, b in pairs:
        keypoints[a], keypoints[b] = keypoints[b], keypoints[a]
    return keypoints


class KeypointFlipd(mt.MapTransform):
    def __init__(
        self,
        keys: list[str],
        pairs: tuple[tuple[int, int], ...],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.pairs = pairs

    def __call__(
        self, data: Mapping[Hashable, Sequence[float]]
    ) -> dict[Hashable, Sequence[float]]:
        d = dict(data)

        for k in self.key_iterator(d):
            d[k] = flip_keypoints(d[k], self.pairs)

        return d
