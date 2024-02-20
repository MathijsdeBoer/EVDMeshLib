"""
3D vectors.
"""

class Vec3:
    """
    A class for 3D vectors.

    Attributes
    ----------
    x : float
        The x component
    y : float
        The y component
    z : float
        The z component
    """

    x: float
    y: float
    z: float

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """
        Create a new vector.

        Attributes
        ----------
        x : float
            The x component
        y : float
            The y component
        z : float
            The z component
        """
    @staticmethod
    def zero() -> "Vec3":
        """
        Returns a new Vec3 instance with all components set to zero.

        Returns
        -------
        Vec3
            A new Vec3 instance with all components set to zero
        """
    @staticmethod
    def one() -> "Vec3":
        """
        Returns a new Vec3 instance with all components set to one.

        Returns
        -------
        Vec3
            A new Vec3 instance with all components set to one
        """
    @staticmethod
    def cartesian_to_spherical(v: "Vec3", wrap_around: bool = False) -> "Vec3":
        """
        Converts cartesian coordinates to spherical coordinates.

        Parameters
        ----------
        v : Vec3
            The Vec3 instance representing cartesian coordinates
        wrap_around : bool, optional
            Whether to wrap the azimuthal angle around the interval [0, 2π]

        Returns
        -------
        Vec3
            A Vec3 instance representing spherical coordinates
        """
    @staticmethod
    def spherical_to_cartesian(vec: "Vec3") -> "Vec3":
        """
        Converts spherical coordinates to cartesian coordinates.

        Parameters
        ----------
        vec : Vec3
            The Vec3 instance representing spherical coordinates

        Returns
        -------
        Vec3
            A Vec3 instance representing cartesian coordinates
        """
    @staticmethod
    def spherical_to_image(v: "Vec3", width: int = 1, height: int = 1) -> tuple[float, float]:
        """
        Converts spherical coordinates to image coordinates.

        Parameters
        ----------
        v : Vec3
            The Vec3 instance representing spherical coordinates
        width : int, optional
            The width of the image
        height : int, optional
            The height of the image

        Returns
        -------
        tuple[float, float]
            A tuple of floats representing image coordinates
        """
    @staticmethod
    def image_to_spherical(x: float, y: float, width: int = 1, height: int = 1) -> "Vec3":
        """
        Converts image coordinates to spherical coordinates.

        Parameters
        ----------
        x : float
            The x coordinate of the image
        y : float
            The y coordinate of the image
        width : int, optional
            The width of the image
        height : int, optional
            The height of the image

        Returns
        -------
        Vec3
            A Vec3 instance representing spherical coordinates
        """
    @staticmethod
    def cartesian_to_cylindrical(vec: "Vec3") -> "Vec3":
        """
        Converts cartesian coordinates to cylindrical coordinates.

        Parameters
        ----------
        vec : Vec3
            The Vec3 instance representing cartesian coordinates

        Returns
        -------
        Vec3
            A Vec3 instance representing cylindrical coordinates
        """
    @staticmethod
    def cylindrical_to_cartesian(vec: "Vec3") -> "Vec3":
        """
        Converts cylindrical coordinates to cartesian coordinates.

        Parameters
        ----------
        vec : Vec3
            The Vec3 instance representing cylindrical coordinates

        Returns
        -------
        Vec3
            A Vec3 instance representing cartesian coordinates
        """
    @property
    def squared_length(self) -> float:
        """
        Returns the squared length of the vector.

        Returns
        -------
        float
            The squared length of the vector
        """
    @property
    def length(self) -> float:
        """
        Returns the length of the vector.

        Returns
        -------
        float
            The length of the vector
        """
    @property
    def unit_vector(self) -> "Vec3":
        """
        Returns a unit vector in the same direction as the original vector.

        Returns
        -------
        Vec3
            A unit vector in the same direction as the original vector
        """
    @property
    def phi(self) -> float:
        """
        Returns the azimuthal angle in spherical coordinates.

        Returns
        -------
        float
            The azimuthal angle
        """
    @property
    def theta(self) -> float:
        """
        Returns the polar angle in spherical coordinates.

        Returns
        -------
        float
            The polar angle
        """
    @property
    def rho(self) -> float:
        """
        Returns the radial distance in spherical coordinates.

        Returns
        -------
        float
            The radial distance
        """
    def dot(self, other: "Vec3") -> float:
        """
        Returns the dot product of this vector and another vector.

        Parameters
        ----------
        other : Vec3
            The other Vec3 instance

        Returns
        -------
        float
            The dot product
        """
    def cross(self, other: "Vec3") -> "Vec3":
        """
        Returns the cross product of this vector and another vector.

        Parameters
        ----------
        other : Vec3
            The other Vec3 instance

        Returns
        -------
        Vec3
            The cross product
        """
    def rotate_around(self, axis: "Vec3", theta: float) -> "Vec3":
        """
        Rotates the vector around a given axis by a given angle.

        Parameters
        ----------
        axis : Vec3
            The axis of rotation
        theta : float
            The angle of rotation

        Returns
        -------
        Vec3
            The rotated vector
        """
    def as_float_list(self) -> list[float]:
        """
        Returns the vector's components as a list of floats.

        Returns
        -------
        list[float]
            The vector's components as a list of floats
        """
    def __add__(self, other: "Vec3") -> "Vec3": ...
    def __iadd__(self, other: "Vec3") -> None: ...
    def __sub__(self, other: "Vec3") -> "Vec3": ...
    def __isub__(self, other: "Vec3") -> None: ...
    def __mul__(self, factor: float) -> "Vec3": ...
    def __imul__(self, factor: float) -> None: ...
    def __truediv__(self, factor: float) -> "Vec3": ...
    def __itruediv__(self, factor: float) -> None: ...
    def __neg__(self) -> "Vec3": ...
    def __eq__(self, other: "Vec3") -> bool: ...
    def __ne__(self, other: "Vec3") -> bool: ...
    def __getitem__(self, index: int) -> float: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
