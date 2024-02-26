"""
Project a mesh to a map.
"""
from math import pi
from pathlib import Path

import click


@click.group(name="project")
@click.pass_context
@click.argument(
    "mesh",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "-r",
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
    count=True,
    help="Increase verbosity (can be specified multiple times)",
)
@click.option(
    "-k",
    "--keypoints",
    "keypoints_file",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    required=False,
    default=None,
    show_default=True,
    help="Name of the keypoints file.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    show_default=True,
    help="Strict mode.",
)
def project_mesh(
    ctx: click.Context,
    mesh: Path,
    output: Path,
    resolution: int,
    gpu: bool,
    verbose: int,
    keypoints_file: Path = None,
    strict: bool = False,
) -> None:
    """
    Project a mesh to a map.

    Parameters
    ----------
    ctx : click.Context
        The click context.
    mesh : Path
        The path to the mesh file.
    output : Path
        The path to the output directory.
    resolution : int
        The resolution of the output image.
    gpu : bool
        Whether to use GPU rendering or not.
    verbose : int
        The verbosity level.
    keypoints_file : Path, optional
        The path to the keypoints file.
    strict : bool
        Whether to use strict mode or not.

    Returns
    -------
    None
    """
    import numpy as np

    from evdplanner.cli import set_verbosity
    from evdplanner.geometry import Mesh

    np.seterr(all="raise")
    set_verbosity(verbose)

    if ctx.obj is None:
        ctx.obj = {}

    ctx.obj["mesh_name"] = mesh.stem.split("_")[1]
    ctx.obj["mesh"] = Mesh.load(str(mesh), 10_000_000)
    ctx.obj["output"] = output
    ctx.obj["resolution"] = resolution
    ctx.obj["gpu"] = gpu
    ctx.obj["verbose"] = verbose
    ctx.obj["keypoints_file"] = keypoints_file
    ctx.obj["strict"] = strict


@project_mesh.command(name="equirectangular")
@click.pass_context
@click.option("--theta-offset", type=float, default=0.5 * pi, show_default=True)
def equirectangular(
    ctx: click.Context,
    theta_offset: float = 0.5 * pi,
) -> None:
    """Project mesh to equirectangular map."""
    import json
    from time import time

    import numpy as np
    from imageio.v3 import imwrite
    from loguru import logger

    from evdplanner.linalg import Vec3
    from evdplanner.markups import MarkupManager, MarkupTypes
    from evdplanner.rendering import Camera, CameraType, IntersectionSort
    from evdplanner.rendering.utils import normalize_image

    logger.info(
        f"Mesh has {ctx.obj['mesh'].num_vertices} vertices and "
        f"{ctx.obj['mesh'].num_triangles} faces"
    )
    logger.debug(f"Mesh origin: {ctx.obj['mesh'].origin}")
    logger.debug(f"Theta offset: {theta_offset}")
    logger.debug(f"Output resolution: {ctx.obj['resolution']}x{ctx.obj['resolution'] // 2}")

    logger.info("Initializing camera...")
    camera = Camera(
        ctx.obj["mesh"].origin,
        forward=Vec3(0, -1, 0),
        up=Vec3(0, 0, 1),
        x_resolution=ctx.obj["resolution"],
        y_resolution=ctx.obj["resolution"] // 2,
        camera_type=CameraType.Equirectangular,
    )

    if ctx.obj["gpu"]:
        from evdplanner.rendering.gpu import GPURenderer

        logger.info("Using GPU renderer")
        renderer = GPURenderer(
            camera,
            mesh=ctx.obj["mesh"],
        )
    else:
        from evdplanner.rendering.cpu import CPURenderer

        logger.info("Using CPU renderer")
        renderer = CPURenderer(
            camera,
            mesh=ctx.obj["mesh"],
        )

    logger.info("Rendering...")
    start = time()
    render = renderer.render(
        intersection_mode=IntersectionSort.Farthest,
    )
    logger.info(f"Rendering took {time() - start:.2f} seconds")

    logger.info("Normalizing images...")
    depth_image = render[..., 0]
    normal_image = render[..., 1:]

    logger.debug(f"Depth image: {depth_image.shape} {depth_image.dtype}")
    logger.debug(f"Normal image: {normal_image.shape} {normal_image.dtype}")

    logger.debug(f"Depth image min: {depth_image.min()}")
    logger.debug(f"Depth image max: {depth_image.max()}")
    logger.debug(f"Normal image min: {normal_image.min()}")
    logger.debug(f"Normal image max: {normal_image.max()}")

    logger.debug("Normalizing images...")
    depth_image = normalize_image(depth_image)
    normal_image += 1.0
    normal_image /= 2.0

    logger.debug(f"Depth image min: {depth_image.min()}")
    logger.debug(f"Depth image max: {depth_image.max()}")
    logger.debug(f"Normal image min: {normal_image.min()}")
    logger.debug(f"Normal image max: {normal_image.max()}")

    logger.debug("Converting images to uint16 and uint8...")
    depth_image = (depth_image * 65535).astype(np.uint16)
    normal_image = (normal_image * 255).astype(np.uint8)

    depth_output = ctx.obj["output"] / f"map_{ctx.obj['mesh_name']}_depth.png"
    normal_output = ctx.obj["output"] / f"map_{ctx.obj['mesh_name']}_normal.png"

    logger.info(f"Saving to {depth_output}")
    imwrite(depth_output, depth_image)

    logger.info(f"Saving to {normal_output}")
    imwrite(normal_output, normal_image)

    logger.info("Saved images")

    if ctx.obj["keypoints_file"]:
        logger.info(f"Projecting keypoints from {ctx.obj['keypoints_file']}")

        if not ctx.obj["keypoints_file"].exists():
            if ctx.obj["strict"]:
                msg = f"File {ctx.obj['keypoints_file']} does not exist."
                logger.error(msg)
                raise FileNotFoundError(msg)
            else:
                logger.warning(
                    f"File {ctx.obj['keypoints_file']} does not exist. "
                    f"Skipping projection of keypoints."
                )
                return

        projected_keypoints: list[dict[str, tuple[float, float]]] = []

        logger.info(f"Loading keypoints from {ctx.obj['keypoints_file']}")
        markups = MarkupManager.load(ctx.obj["keypoints_file"])

        logger.info("Projecting keypoints...")
        for markup in markups.markups:
            if markup.markup_type == MarkupTypes.FIDUCIAL:
                for point in markup.control_points:
                    if len(point.position) == 0:
                        if not ctx.obj["strict"]:
                            logger.warning(f"Empty position for {point.label}")
                            continue
                        else:
                            msg = f"Empty position for {point.label}"
                            logger.error(msg)
                            raise ValueError(msg)

                    control_point = Vec3(*point.position[:3])
                    x, y = camera.project_back(control_point, normalized=True)

                    logger.debug(f"Projected {point.label}: {str(control_point)} -> ({x=}, {y=})")

                    projected_keypoints.append(
                        {
                            "label": point.label,
                            "position": (x, y),
                        }
                    )

        output_file = ctx.obj["output"] / f"projected_{ctx.obj['mesh_name']}.kp.json"
        logger.info(f"Saving projected keypoints to {output_file}")

        with output_file.open("w") as f:
            json.dump(projected_keypoints, f, indent=4)
