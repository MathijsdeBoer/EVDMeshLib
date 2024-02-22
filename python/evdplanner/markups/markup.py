"""
The Markup class, which is used to represent a markup in the 3D Slicer scene.
"""

from dataclasses import dataclass, field
from enum import Enum

from .display_settings import DisplaySettings
from .fiducial import Fiducial


class MarkupTypes(Enum):
    """
    Enum class for the different types of markups that can be created in 3D Slicer.

    The different types are:
    - ANGLE: An angle markup
    - CURVE: A curve markup
    - FIDUCIAL: A fiducial markup
    - LINE: A line markup
    """

    ANGLE: str = "Angle"
    CURVE: str = "Curve"
    FIDUCIAL: str = "Fiducial"
    LINE: str = "Line"


@dataclass
class Markup:
    """
    Class to represent a markup in the 3D Slicer scene.

    Attributes
    ----------
    markup_type : MarkupTypes
        The type of markup to create.
    display_settings : DisplaySettings
        The display settings for the markup.
    control_points : list[Fiducial]
        The control points for the markup.
    """

    markup_type: MarkupTypes = MarkupTypes.FIDUCIAL
    display_settings: DisplaySettings = field(default_factory=DisplaySettings.default)
    control_points: list[Fiducial] = field(default_factory=list)

    def add_control_point(
        self, label: str, description: str, position: list[float], visible: bool = True
    ) -> None:
        """
        Add a control point to the markup.

        Parameters
        ----------
        label : str
            The label for the control point.
        description : str
            The description for the control point.
        position : list[float]
            The position of the control point.
        visible : bool
            Whether the control point is visible or not.

        Returns
        -------
        None
        """
        self.control_points.append(
            Fiducial(
                markup_id=len(self.control_points) + 1,
                label=label,
                description=description,
                position=position,
                visible=visible,
            )
        )

    @staticmethod
    def from_dict(data: dict) -> "Markup":
        """
        Create a Markup object from a dictionary.

        Parameters
        ----------
        data : dict

        Returns
        -------
        Markup
            The markup object created from the dictionary.
        """
        return Markup(
            markup_type=MarkupTypes(data["type"]),
            display_settings=DisplaySettings.from_dict(data["display"]),
            control_points=[Fiducial.from_dict(cp) for cp in data["controlPoints"]],
        )

    def to_dict(self) -> dict:
        """
        Convert the markup to a dictionary.

        Returns
        -------
        dict
            The dictionary representation of the markup.
        """
        return {
            "type": self.markup_type.value,
            "coordinateSystem": "LPS",
            "coordinateUnits": "mm",
            "locked": True,
            "fixedNumberOfControlPoints": True,
            "labelFormat": "%N-%d",
            "lastUsedControlPointNumber": len(self.control_points),
            "controlPoints": [cp.to_dict() for cp in self.control_points],
            "measurements": [],
            "display": self.display_settings.to_dict(),
        }
