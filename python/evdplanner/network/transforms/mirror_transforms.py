"""
Mirror Transform.
"""
import monai.transforms as mt


class MirrorTransform(mt.MapTransform, mt.InvertibleTransform):
    """
    Mirror the image and the label.
    """

    def __init__(
        self, keys: list[str], image_key: str = "image", label_key: str = "label"
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
        raise NotImplementedError
