from evdplanner.rs.linalg import Vec3

class Deformer:
    def __init__(
        self,
        scale: float,
        amplitude: float,
        frequency: float,
        octaves: int,
        persistence: float,
        lacunarity: float,
        seed: int,
    ) -> None:
        """
        Initialize the Deformer with the given parameters.

        Parameters
        ----------
        scale : float
            The scale of the deformation.
        amplitude : float
            The amplitude of the deformation.
        frequency : float
            The frequency of the deformation.
        octaves : int
            The number of octaves in the deformation.
        persistence : float
            The persistence of the deformation.
        lacunarity : float
            The lacunarity of the deformation.
        seed : int
            The seed of the deformation.
        """

    def deform_vertex(self, vertex: Vec3) -> Vec3:
        """
        Deform the given vertex.

        Parameters
        ----------
        vertex : Vec3
            The vertex to deform.

        Returns
        -------
        Vec3
            The deformed vertex.
        """
