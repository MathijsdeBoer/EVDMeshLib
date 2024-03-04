"""
The Fiducial class, which is used to represent a single fiducial in the Slicer scene.
"""

from dataclasses import dataclass


@dataclass
class Fiducial:
    """
    A class to represent a single fiducial in the Slicer scene.

    Attributes
    ----------
    markup_id : int
        The unique identifier of the fiducial.
    label : str
        The label of the fiducial.
    description : str
        The description of the fiducial.
    position : list[float] | None
        The position of the fiducial in the 3D space. If None, the position is undefined.
    visible : bool
        Whether the fiducial is visible in the 3D view.
    """

    markup_id: int
    label: str
    description: str
    position: list[float] | None = None
    locked: bool = True
    visible: bool = True

    @staticmethod
    def from_dict(data: dict) -> "Fiducial":
        """
        Create a Fiducial object from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing the properties of the fiducial.

        Returns
        -------
        Fiducial
            A Fiducial object with the properties specified in the input dictionary.
        """
        return Fiducial(
            markup_id=data["id"],
            label=data["label"],
            description=data["description"],
            position=data["position"],
            locked=data["locked"],
            visible=data["visibility"],
        )

    def to_dict(self) -> dict:
        """
        Convert the Fiducial object to a dictionary.

        Returns
        -------
        dict
            A dictionary containing the properties of the fiducial.
        """
        return {
            "id": str(self.markup_id),
            "label": self.label,
            "description": self.description,
            "associatedNodeID": "",
            "position": self.position or [],
            "orientation": [
                -1.0,
                -0.0,
                -0.0,
                -0.0,
                -1.0,
                -0.0,
                0.0,
                0.0,
                1.0,
            ],
            "selected": True,
            "locked": self.locked,
            "visibility": self.visible,
            "positionStatus": "defined" if self.position else "undefined",
        }
