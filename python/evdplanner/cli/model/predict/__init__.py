from pathlib import Path

import click

from evdplanner.network import PointRegressor


@click.command(name="predict-skin-mesh")
def predict_skin_mesh(
    model_path: Path,
    mesh: Path,
    output_path: Path,
    gpu_renderer: bool = False,
    gpu_model: bool = False,
    verbose: bool = False,
):
    import torch

    from evdplanner.geometry import Mesh
    from evdplanner.linalg import Vec3
    from evdplanner.rendering import Camera, CameraType, IntersectionSort
    if gpu_renderer:
        if verbose:
            print("Using GPU renderer.")
        from evdplanner.rendering.gpu import GPURenderer as Renderer
    else:
        if verbose:
            print("Using CPU renderer.")
        from evdplanner.rendering import CPURenderer as Renderer

    if verbose:
        print("Loading model...")
    device = torch.device("cuda" if gpu_model else "cpu")
    model: PointRegressor = torch.load(model_path, map_location=device)
    model.eval()

    resolution = model.in_shape

    if verbose:
        print(f"Model resolution: {resolution}")
        print("Loading mesh...")
    mesh = Mesh.load(str(mesh), num_samples=10_000_000)

    if verbose:
        print("Creating camera...")
    camera = Camera(
        origin=mesh.origin,
        forward=Vec3(0.0, -1.0, 0.0),
        up=Vec3(0.0, 0.0, 1.0),
        x_resolution=resolution[2],
        y_resolution=resolution[1],
        camera_type=CameraType.Equirectangular,
    )

    renderer = Renderer(
        camera=camera,
        mesh=mesh,
    )

    if verbose:
        print("Rendering...")
    image = renderer.render(IntersectionSort.Farthest)

    if verbose:
        print("Predicting...")
    with torch.no_grad():
        prediction = model(image).squeeze().cpu().numpy()

    if verbose:
        print("Saving...")


