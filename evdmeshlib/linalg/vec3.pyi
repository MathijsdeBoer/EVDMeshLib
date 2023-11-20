class Vec3:
    x: float
    y: float
    z: float

    def __init__(self, x=0.0, y=0.0, z=0.0):
        ...

    @staticmethod
    def zero() -> Vec3:
        ...

    @staticmethod
    def one() -> Vec3:
        ...

    @staticmethod
    def cartesian_to_spherical(vec: Vec3, wrap_around: bool = False) -> Vec3:
        ...

    @staticmethod
    def spherical_to_cartesian(vec: Vec3) -> Vec3:
        ...

    @staticmethod
    def spherical_to_image(vec: Vec3, width: int = 1, height: int = 1) -> Vec3:
        ...

    @staticmethod
    def image_to_spherical(x: float, y: float, width: int = 1, height: int = 1) -> Vec3:
        ...

    @staticmethod
    def cartesian_to_cylindrical(vec: Vec3) -> Vec3:
        ...

    @staticmethod
    def cylindrical_to_cartesian(vec: Vec3) -> Vec3:
        ...

    @property
    def squared_length(self) -> float:
        ...

    @property
    def length(self) -> float:
        ...

    @property
    def unit_vector(self) -> Vec3:
        ...

    @property
    def phi(self) -> float:
        ...

    @property
    def theta(self) -> float:
        ...

    @property
    def rho(self) -> float:
        ...

    def dot(self, other: Vec3) -> float:
        ...

    def cross(self, other: Vec3) -> Vec3:
        ...

    def rotate_around_x(self, angle: float) -> Vec3:
        ...

    def as_float_list(self) -> list[float]:
        ...

    def __add__(self, other: Vec3) -> Vec3:
        ...

    def __iadd__(self, other: Vec3) -> Vec3:
        ...

    def __sub__(self, other: Vec3) -> Vec3:
        ...

    def __isub__(self, other: Vec3) -> Vec3:
        ...

    def __mul__(self, other: float) -> Vec3:
        ...

    def __imul__(self, other: float) -> Vec3:
        ...

    def __truediv__(self, other: float) -> Vec3:
        ...

    def __itruediv__(self, other: float) -> Vec3:
        ...

    def __neg__(self) -> Vec3:
        ...

    def __eq__(self, other: Vec3) -> bool:
        ...

    def __ne__(self, other: Vec3) -> bool:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...
