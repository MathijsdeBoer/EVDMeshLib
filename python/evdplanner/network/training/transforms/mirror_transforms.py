import monai.transforms as mt


class MirrorTransform(mt.MapTransform, mt.InvertibleTransform):
    def __init__(self, keys: list[str], image_key: str = "image", label_key: str = "label"):
        super().__init__(keys)
        self.image_key = image_key
        self.label_key = label_key

    def __call__(self, data):
        raise NotImplementedError
