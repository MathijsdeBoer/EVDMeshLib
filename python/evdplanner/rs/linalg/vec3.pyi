class Vec3:
    x: float
    y: float
    z: float

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """
        Create a new vector.

        :param x: The x component
        :param y: The y component
        :param z: The z component
        """
    @staticmethod
    def zero() -> "Vec3":
        """
        Returns a new Vec3 instance with all components set to zero.

        :return: A new Vec3 instance
        """
    @staticmethod
    def one() -> "Vec3":
        """
        Returns a new Vec3 instance with all components set to one.

        :return: A new Vec3 instance
        """
    @staticmethod
    def cartesian_to_spherical(v: "Vec3", wrap_around: bool = False) -> "Vec3":
        """
        Converts cartesian coordinates to spherical coordinates.

        :param v: The Vec3 instance representing cartesian coordinates
        :param wrap_around: A boolean indicating whether to wrap around
        :return: A Vec3 instance representing spherical coordinates
        """
    @staticmethod
    def spherical_to_cartesian(vec: "Vec3") -> "Vec3":
        """
        Converts spherical coordinates to cartesian coordinates.

        :param vec: The Vec3 instance representing spherical coordinates
        :return: A Vec3 instance representing cartesian coordinates
        """
    @staticmethod
    def spherical_to_image(v: "Vec3", width: int = 1, height: int = 1) -> tuple[float, float]:
        """
        Converts spherical coordinates to image coordinates.

        :param v: The Vec3 instance representing spherical coordinates
        :param width: The width of the image
        :param height: The height of the image
        :return: A tuple representing image coordinates
        """
    @staticmethod
    def image_to_spherical(x: float, y: float, width: int = 1, height: int = 1) -> "Vec3":
        """
        Converts image coordinates to spherical coordinates.

        :param x: The x coordinate of the image
        :param y: The y coordinate of the image
        :param width: The width of the image
        :param height: The height of the image
        :return: A Vec3 instance representing spherical coordinates
        """
    @staticmethod
    def cartesian_to_cylindrical(vec: "Vec3") -> "Vec3":
        """
        Converts cartesian coordinates to cylindrical coordinates.

        :param vec: The Vec3 instance representing cartesian coordinates
        :return: A Vec3 instance representing cylindrical coordinates
        """
    @staticmethod
    def cylindrical_to_cartesian(vec: "Vec3") -> "Vec3":
        """
        Converts cylindrical coordinates to cartesian coordinates.

        :param vec: The Vec3 instance representing cylindrical coordinates
        :return: A Vec3 instance representing cartesian coordinates
        """
    @property
    def squared_length(self) -> float:
        """
        Returns the squared length of the vector.

        :return: The squared length of the vector
        """
    @property
    def length(self) -> float:
        """
        Returns the length of the vector.

        :return: The length of the vector
        """
    @property
    def unit_vector(self) -> "Vec3":
        """
        Returns a unit vector in the same direction as the original vector.

        :return: A unit vector
        """
    @property
    def phi(self) -> float:
        """
        Returns the azimuthal angle in spherical coordinates.

        :return: The azimuthal angle
        """
    @property
    def theta(self) -> float:
        """
        Returns the polar angle in spherical coordinates.

        :return: The polar angle
        """
    @property
    def rho(self) -> float:
        """
        Returns the radial distance in spherical coordinates.

        :return: The radial distance
        """
    def dot(self, other: "Vec3") -> float:
        """
        Returns the dot product of this vector and another vector.

        :param other: The other Vec3 instance
        :return: The dot product
        """
    def cross(self, other: "Vec3") -> "Vec3":
        """
        Returns the cross product of this vector and another vector.

        :param other: The other Vec3 instance
        :return: The cross product
        """
    def rotate_around(self, axis: "Vec3", theta: float) -> "Vec3":
        """
        Rotates the vector around a given axis by a given angle.

        :param axis: The axis to rotate around
        :param theta: The angle to rotate by
        :return: The rotated vector
        """
    def as_float_list(self) -> list[float]:
        """
        Returns the vector's components as a list of floats.

        :return: A list of floats representing the vector's components
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
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
