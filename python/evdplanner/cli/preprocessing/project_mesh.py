"""
Project a mesh to a map.
"""

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
    from loguru import logger

    from evdplanner.cli import set_verbosity
    from evdplanner.geometry import Mesh

    np.seterr(all="raise")
    set_verbosity(verbose)

    if ctx.obj is None:
        ctx.obj = {}

    ctx.obj["mesh_name"] = mesh.stem.split("_")[1]
    logger.info("Loading mesh...")
    ctx.obj["mesh"] = Mesh.load(str(mesh), 10_000_000)
    ctx.obj["output"] = output
    ctx.obj["resolution"] = resolution
    ctx.obj["keypoints_file"] = keypoints_file
    ctx.obj["strict"] = strict


@project_mesh.command(name="equirectangular")
@click.pass_context
def equirectangular(
    ctx: click.Context,
) -> None:
    """Project mesh to equirectangular map."""
    import json
    from time import time

    import numpy as np
    from imageio.v3 import imwrite
    from loguru import logger

    from evdplanner.linalg import Vec3
    from evdplanner.markups import MarkupManager, MarkupTypes
    from evdplanner.rendering import Camera, CameraType, IntersectionSort, Renderer
    from evdplanner.rendering.utils import normalize_image

    logger.debug(f"Number of triangles: {ctx.obj['mesh'].num_triangles}")
    logger.debug(f"Mesh origin: {ctx.obj['mesh'].origin}")
    logger.debug(f"Output resolution: {ctx.obj['resolution']}x{ctx.obj['resolution'] // 2}")

    mesh = ctx.obj["mesh"]
    logger.info("Flattening BVH...")
    mesh.flatten_bvh()

    logger.info("Initializing camera...")
    camera = Camera(
        mesh.origin,
        forward=Vec3(0, -1, 0),
        up=Vec3(0, 0, 1),
        x_resolution=ctx.obj["resolution"],
        y_resolution=ctx.obj["resolution"] // 2,
        camera_type=CameraType.Equirectangular,
    )

    renderer = Renderer(
        camera,
        mesh=mesh,
    )

    logger.info("Rendering...")
    start = time()
    render = renderer.render(
        intersection_mode=IntersectionSort.Farthest,
    )
    end = time()
    delta = end - start
    logger.info(f"Rendering took {delta:.2f} seconds")

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

    if not ctx.obj["output"].exists():
        ctx.obj["output"].mkdir(parents=True)

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


@project_mesh.command(name="perspective")
@click.pass_context
@click.option(
    "-k",
    "--kocher",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="Path to the Kocher projection file.",
)
@click.option(
    "-s",
    "--side",
    type=click.Choice(["left", "right"]),
    default=None,
    show_default=True,
    help="Side of the Kocher projection.",
)
def perspective(
    ctx: click.Context,
    kocher: Path,
    side: str | None = None,
) -> None:
    import numpy as np
    from imageio.v3 import imwrite
    from loguru import logger

    from evdplanner.linalg import Vec3
    from evdplanner.markups import MarkupManager
    from evdplanner.rendering import Camera, CameraType, IntersectionSort, Renderer
    from evdplanner.rendering.utils import normalize_image

    logger.info(f"Loading Kocher projection from {kocher}")
    markups = MarkupManager.load(kocher)
    logger.debug("Searching for Kocher's Points")
    left_kp = markups.find_fiducial("Left Kocher")
    right_kp = markups.find_fiducial("Right Kocher")
    logger.debug(f"Left Kocher: {left_kp}")
    logger.debug(f"Right Kocher: {right_kp}")

    if left_kp is None or right_kp is None:
        msg = f"Could not find Kocher points in {kocher}"
        logger.error(msg)
        raise ValueError(msg)

    left_kp = Vec3(*left_kp.position[:3])
    right_kp = Vec3(*right_kp.position[:3])

    sides_to_render: list[tuple[Vec3, str]] = []
    if not side:
        sides_to_render = [(left_kp, "left"), (right_kp, "right")]
    elif side == "left":
        sides_to_render.append((left_kp, "left"))
    elif side == "right":
        sides_to_render.append((right_kp, "right"))
    else:
        msg = f"Invalid side: {side}"
        logger.error(msg)
        raise ValueError(msg)

    mesh = ctx.obj["mesh"]
    renders: list[tuple[str, np.ndarray]] = []
    for kp, side in sides_to_render:
        logger.info(f"Projecting mesh to {side} Kocher's Point")
        camera = Camera(
            kp,
            forward=(mesh.origin - kp).unit_vector,
            up=Vec3(0, 0, -1),
            x_resolution=ctx.obj["resolution"],
            y_resolution=ctx.obj["resolution"],
            camera_type=CameraType.Perspective,
            fov=90,
            size=125,
        )
        renderer = Renderer(
            camera,
            mesh=mesh,
        )

        logger.info("Rendering...")
        far_render = renderer.render(
            intersection_mode=IntersectionSort.Farthest,
        )
        near_render = renderer.render(
            intersection_mode=IntersectionSort.Nearest,
        )
        logger.info("Rendering done")

        render = near_render
        thickness = far_render[..., 0] - near_render[..., 0]
        # Add thickness to the first channel
        render = np.dstack((thickness, render))

        renders.append((side, render))

    for side, render in renders:
        thickness_image = render[..., 0]
        depth_image = render[..., 1]
        normal_image = render[..., 2:]

        logger.debug(f"Thickness image: {thickness_image.shape} {thickness_image.dtype}")
        logger.debug(f"Thickness Image: {thickness_image.min()} {thickness_image.max()}")
        logger.debug(f"Depth image: {depth_image.shape} {depth_image.dtype}")
        logger.debug(f"Depth Image: {depth_image.min()} {depth_image.max()}")
        logger.debug(f"Normal image: {normal_image.shape} {normal_image.dtype}")
        logger.debug(f"Normal Image: {normal_image.min()} {normal_image.max()}")

        thickness_image = normalize_image(thickness_image)
        depth_image = normalize_image(depth_image)
        normal_image += 1.0
        normal_image /= 2.0

        thickness_image = (thickness_image * 65535).astype(np.uint16)
        depth_image = (depth_image * 65535).astype(np.uint16)
        normal_image = (normal_image * 255).astype(np.uint8)

        if not ctx.obj["output"].exists():
            ctx.obj["output"].mkdir(parents=True)

        thickness_output = ctx.obj["output"] / f"map_{ctx.obj['mesh_name']}_{side}_thickness.png"
        depth_output = ctx.obj["output"] / f"map_{ctx.obj['mesh_name']}_{side}_depth.png"
        normal_output = ctx.obj["output"] / f"map_{ctx.obj['mesh_name']}_{side}_normal.png"

        logger.info(f"Saving to {thickness_output}")
        imwrite(thickness_output, thickness_image)
        logger.info(f"Saving to {depth_output}")
        imwrite(depth_output, depth_image)
        logger.info(f"Saving to {normal_output}")
        imwrite(normal_output, normal_image)
