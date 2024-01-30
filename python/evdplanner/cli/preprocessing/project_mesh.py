from math import pi
from pathlib import Path
from time import time

import click
import numpy as np
from imageio import imwrite

from evdplanner.rendering.utils import normalize_image
from evdplanner.rs import Mesh, Camera, Vec3, CameraType, IntersectionSort


class _ProjectionSession:
    def __init(
            self,
            mesh_name: str,
            mesh: Mesh,
            output: Path,
            resolution: int,
            gpu: bool,
            verbose: bool,
    ):
        self.mesh_name = mesh_name
        self.mesh = mesh
        self.output = output
        self.resolution = resolution
        self.gpu = gpu
        self.verbose = verbose


@click.group(name="project")
@click.pass_context
@click.argument(
    "mesh",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, writable=True, resolve_path=True, path_type=Path
    ),
)
@click.option(
    "--resolution",
    type=int,
    default=1024,
    show_default=True,
    help="Resolution of the output image.",
)
@click.option(
    "--gpu",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use GPU rendering.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    show_default=True,
    help="Print progress.",
)
def project_mesh(
        ctx: click.Context,
        mesh: Path,
        output: Path,
        resolution: int,
        gpu: bool,
        verbose: bool,
):
    if ctx.obj is None:
        ctx.obj = {}

    ctx.obj["mesh_name"] = mesh.stem.split("_")[-1]
    ctx.obj["mesh"] = Mesh.load(str(mesh), 10_000_000)
    ctx.obj["output"] = output
    ctx.obj["resolution"] = resolution
    ctx.obj["gpu"] = gpu
    ctx.obj["verbose"] = verbose


@project_mesh.command(name="equirectangular")
@click.pass_context
@click.option("--theta-offset", type=float, default=0.5 * pi, show_default=True)
def equirectangular(
        ctx: click.Context,
        theta_offset: float = 0.5 * pi,
):
    """Project mesh to equirectangular map."""
    if ctx.obj["verbose"]:
        print(f"Mesh has {ctx.obj['mesh'].num_vertices} vertices and {ctx.obj['mesh'].num_triangles} faces")
        print(f"Mesh origin: {ctx.obj['mesh'].origin}")
        print(f"Theta offset: {theta_offset}")
        print(f"Output resolution: {ctx.obj['resolution']}x{ctx.obj['resolution'] // 2}")
        print("Initializing camera...")

    camera = Camera(
        ctx.obj["mesh"].origin,
        forward=Vec3(0, -1, 0),
        up=Vec3(0, 0, 1),
        x_resolution=ctx.obj["resolution"],
        y_resolution=ctx.obj["resolution"] // 2,
        camera_type=CameraType.Equirectangular,
        theta_offset=theta_offset,
    )

    if ctx.obj["gpu"]:
        from evdplanner.rendering.gpu import GPURenderer

        if ctx.obj["verbose"]:
            print("Using GPU renderer")

        renderer = GPURenderer(
            camera,
            mesh=ctx.obj["mesh"],
        )
    else:
        from evdplanner.rendering.cpu import CPURenderer

        if ctx.obj["verbose"]:
            print("Using CPU renderer")

        renderer = CPURenderer(
            camera,
            mesh=ctx.obj["mesh"],
        )

    if ctx.obj["verbose"]:
        print("Rendering...")
        start = time()

    render = renderer.render(
        intersection_mode=IntersectionSort.Farthest,
    )

    if ctx.obj["verbose"]:
        print(f"Rendering took {time() - start:.2f}s")
        print("Saving...")

    depth_image = render[..., 0]
    normal_image = render[..., 1:]

    depth_image = normalize_image(depth_image)
    normal_image += 1.0
    normal_image /= 2.0

    depth_image = (depth_image * 65535).astype(np.uint16)
    normal_image = (normal_image * 255).astype(np.uint8)

    depth_output = ctx.obj["output"] / f"map_{ctx.obj['mesh_name']}_depth.png"
    normal_output = ctx.obj["output"] / f"map_{ctx.obj['mesh_name']}_normal.png"

    imwrite(depth_output, depth_image)
    imwrite(normal_output, normal_image)

    if ctx.obj["verbose"]:
        print(f"Saved to {depth_output}")
        print(f"Saved to {normal_output}")
