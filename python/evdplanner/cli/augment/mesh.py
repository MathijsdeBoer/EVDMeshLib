from math import pi
from pathlib import Path

import click
import opensimplex

from evdplanner.linalg import Vec3


def _generate_noise(
    position: Vec3,
    generator: opensimplex.OpenSimplex,
    amplitude: float,
    frequency: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
) -> Vec3:
    """
    Generate noise at a given position.

    Parameters
    ----------
    position : Vec3
        The position at which to generate the noise.
    amplitude : float
        The amplitude of the noise.
    frequency : float
        The frequency of the noise.
    octaves : int
        The number of octaves.
    persistence : float
        The persistence of the noise.
    lacunarity : float
        The lacunarity of the noise.

    Returns
    -------
    Vec3
        The noise at the given position.
    """
    x = y = z = 0.0

    original_seed = generator.get_seed()
    for i in range(octaves):
        x += amplitude * generator.noise3(
            position.x * frequency, position.y * frequency, position.z * frequency
        )
        y += amplitude * generator.noise3(
            position.y * frequency, position.z * frequency, position.x * frequency
        )
        z += amplitude * generator.noise3(
            position.z * frequency, position.x * frequency, position.y * frequency
        )

        generator._seed = original_seed + i

        amplitude *= persistence
        frequency *= lacunarity
    generator._seed = original_seed

    return Vec3(x, y, z)


@click.command()
@click.argument(
    "input_mesh",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(dir_okay=False, resolve_path=True, path_type=Path),
)
@click.option(
    "--keypoints",
    "keypoints_path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="Path to the keypoints file.",
)
@click.option("--rotate", is_flag=True, help="Rotate the mesh.")
@click.option(
    "--rotate-range",
    type=float,
    default=None,
    required=False,
    help="Range of random rotation in pi radians.",
)
@click.option("--deform", is_flag=True, help="Deform the mesh.")
@click.option(
    "--deform-scale", type=float, default=15.0, required=False, help="Scale of the deformation."
)
@click.option(
    "--deform-amplitude",
    type=float,
    default=1.0,
    required=False,
    help="Range of random deformation.",
)
@click.option(
    "--deform-frequency",
    type=float,
    default=0.001,
    required=False,
    help="Frequency of the noise.",
)
@click.option(
    "--deform-octaves",
    type=int,
    default=6,
    required=False,
    help="Number of octaves.",
)
@click.option(
    "--deform-persistence",
    type=float,
    default=0.5,
    required=False,
    help="Persistence of the noise.",
)
@click.option(
    "--deform-lacunarity",
    type=float,
    default=2.0,
    required=False,
    help="Lacunarity of the noise.",
)
@click.option("-v", "--verbosity", count=True, help="Increase output verbosity.")
@click.option(
    "--seed", type=int, default=None, required=False, help="Seed for the noise generation."
)
def mesh(
    input_mesh: Path,
    output: Path,
    keypoints_path: Path,
    rotate: bool = False,
    rotate_range: float | None = None,
    deform: bool = False,
    deform_scale: float = 15.0,
    deform_amplitude: float = 1.0,
    deform_frequency: float = 0.0025,
    deform_octaves: int = 6,
    deform_persistence: float = 0.5,
    deform_lacunarity: float = 2.0,
    verbosity: int = 0,
    seed: int | None = None,
) -> None:
    """
    Augment data.
    """
    import json
    import random

    from loguru import logger

    from evdplanner.cli import set_verbosity
    from evdplanner.geometry import Deformer, Mesh
    from evdplanner.linalg import Mat4
    from evdplanner.markups import MarkupManager

    set_verbosity(verbosity)

    if not seed:
        seed = random.randint(-1_000_000_000, 1_000_000_000)
        logger.debug(f"Seed not provided. Using random seed {seed}.")
    random.seed(seed)

    if not rotate and not deform:
        logger.warning("No augmentation selected. Exiting.")
        return

    if not rotate and rotate_range:
        logger.warning("--rotate-range set, but --rotate omitted. Ignoring --rotate-range.")
        rotate_range = None
    elif rotate and not rotate_range:
        logger.warning("--rotate set, but --rotate-range omitted. Setting --rotate-range to 1.0.")
        rotate_range = 1.0

    if not deform:
        if deform_scale:
            logger.warning("--deform-scale set, but --deform omitted. Ignoring --deform-scale.")
            deform_scale = 1.0
        if deform_amplitude:
            logger.warning("--deform-range set, but --deform omitted. Ignoring --deform-range.")
            deform_amplitude = None
        if deform_frequency:
            logger.warning(
                "--deform-frequency set, but --deform omitted. Ignoring --deform-frequency."
            )
            deform_frequency = 0.5
        if deform_octaves:
            logger.warning(
                "--deform-octaves set, but --deform omitted. Ignoring --deform-octaves."
            )
            deform_octaves = 3
        if deform_persistence:
            logger.warning(
                "--deform-persistence set, but --deform omitted. Ignoring --deform-persistence."
            )
            deform_persistence = 0.5
        if deform_lacunarity:
            logger.warning(
                "--deform-lacunarity set, but --deform omitted. Ignoring --deform-lacunarity."
            )
            deform_lacunarity = 2.0

    logger.info(f"Loading mesh from {input_mesh}.")
    m = Mesh.load(str(input_mesh))
    logger.debug(f"Mesh origin: {m.origin}.")
    logger.debug(f"Number of triangles: {m.num_triangles}.")

    logger.info(f"Loading keypoints from {keypoints_path}.")
    keypoints = MarkupManager.load(keypoints_path)
    logger.debug(f"Number of markups: {len(keypoints.markups)}.")

    if rotate:
        logger.info(f"Rotating mesh with range {rotate_range}.")

        random_axis = Vec3(
            random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)
        ).unit_vector
        random_angle = random.uniform(-rotate_range * pi, rotate_range * pi)
        logger.debug(f"Random axis: {random_axis}.")
        logger.debug(f"Random angle: {random_angle}.")

        origin = m.origin
        matrix = (
            Mat4.translation(-origin.x, -origin.y, -origin.z)
            * Mat4.rotation(random_axis, random_angle)
            * Mat4.translation(origin.x, origin.y, origin.z)
        )

        logger.debug(f"Rotating mesh vertices.")
        m.transform(matrix)

        logger.debug(f"Rotating keypoints.")
        for mark in keypoints.markups:
            logger.debug(f"Rotating control points for {mark.markup_type}.")
            for point in mark.control_points:
                logger.debug(f"Rotating control point {point.label}.")
                logger.debug(f"Old position: {point.position}.")
                p = Vec3(*point.position)
                p = p @ matrix
                point.position = [p.x, p.y, p.z]
                logger.debug(f"New position: {point.position}.")

    if deform:
        logger.info(f"Deforming mesh with range {deform_amplitude}.")
        logger.debug(f"Deform scale: {deform_scale}.")
        logger.debug(f"Deform amplitude: {deform_amplitude}.")
        logger.debug(f"Deform frequency: {deform_frequency}.")
        logger.debug(f"Deform octaves: {deform_octaves}.")
        logger.debug(f"Deform persistence: {deform_persistence}.")
        logger.debug(f"Deform lacunarity: {deform_lacunarity}.")

        deformer = Deformer(
            scale=deform_scale,
            amplitude=deform_amplitude,
            frequency=deform_frequency,
            octaves=deform_octaves,
            persistence=deform_persistence,
            lacunarity=deform_lacunarity,
            seed=seed,
        )
        m.deform(deformer)

        for mark in keypoints.markups:
            logger.debug(f"Deforming control points for {mark.markup_type}.")
            for point in mark.control_points:
                logger.debug(f"Deforming control point {point.label}.")
                logger.debug(f"Old position: {point.position}.")
                p = Vec3(*point.position)
                noise = deformer.deform_vertex(p)
                p += noise
                point.position = [p.x, p.y, p.z]
                logger.debug(f"New position: {point.position}.")

    if not output.parent.exists():
        output.parent.mkdir(parents=True)

    logger.info(f"Saving mesh to {output}.")
    m.recalculate(1_000_000)
    m.save(str(output))

    keypoint_output = output.parent / keypoints_path.name
    logger.info(f"Saving keypoints to {keypoint_output}.")
    with keypoint_output.open("w") as f:
        json.dump(keypoints.to_dict(), f)

    m.save(str(output))
