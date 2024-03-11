from math import log2
import pytest
from torch import rand

from evdplanner.network.architecture.point_regressor import PointRegressor


@pytest.mark.parametrize("resolution", [256 * 2 ** i for i in range(4)])
@pytest.mark.parametrize("initial_filters", [8 * 2 ** i for i in range(4)])
@pytest.mark.parametrize("num_res_units", list(range(5)))
def test_skin_point_regressor(
    resolution: int,
    initial_filters: int,
    num_res_units: int,
):
    maps = ["one", "two"]
    keypoints = ["one", "two", "three"]

    in_shape = (1, resolution, resolution // 2)
    out_shape = (len(keypoints), 2)

    depth = int(log2(resolution // 16))

    channels = [initial_filters * 2 ** i for i in range(depth)]

    model = PointRegressor(
        maps=maps,
        keypoints=keypoints,
        in_shape=in_shape,
        out_shape=out_shape,
        channels=channels,
        strides=[2] * depth,
        kernel_size=3,
        num_res_units=num_res_units,
    )

    random_input = rand(1, *in_shape)
    output = model(random_input)
    assert output.shape == (1, len(keypoints), 2)
