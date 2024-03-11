from pathlib import Path

import monai.transforms as mt
import numpy as np

from evdplanner.linalg import Vec3
from evdplanner.rs import Camera, CameraType, Mesh


class LandmarksToKeypointsd(mt.MapTransform):
    def __init__(
        self,
        landmarks_keys: str,
        mesh_key: str,
        keypoints_key: str,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys=[mesh_key, landmarks_keys], allow_missing_keys=allow_missing_keys)

        self.landmarks_keys = landmarks_keys
        self.mesh_key = mesh_key
        self.keypoints_key = keypoints_key

    def __call__(self, data: dict[str, Path | Mesh]) -> dict:
        d = dict(data)

        origin = d[self.mesh_key].origin
        landmarks, names = d[self.landmarks_keys]

        camera = Camera(
            origin=origin,
            forward=Vec3(0.0, -1.0, 0.0),
            up=Vec3(0.0, 0.0, 1.0),
            camera_type=CameraType.Equirectangular,
            x_resolution=1,
            y_resolution=1,
        )

        keypoints = []
        for i, landmark in enumerate(landmarks):
            keypoints.append(camera.project_back(landmark))

        d[self.keypoints_key] = np.array(keypoints)

        return d
