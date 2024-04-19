"""
A class to manage markups in 3D Slicer.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from .display_settings import DisplaySettings
from .fiducial import Fiducial
from .markup import Markup, MarkupTypes


class MarkupManager:
    """
    Class to manage markups in 3D Slicer.

    Attributes
    ----------
    markups : list[Markup]
        List of markups.
    """

    markups: list[Markup]

    def __init__(self) -> None:
        """
        Initialize the class.
        """
        self.markups = []

    def find_fiducial(self, label: str) -> Fiducial | None:
        """
        Find a markup by its label.

        Parameters
        ----------
        label : str
            Label of the markup.

        Returns
        -------
        Markup or None
            Markup with the given label or None if not found.
        """
        for markup in self.markups:
            for control_point in markup.control_points:
                if control_point.label == label:
                    return control_point
        return None

    def _add_item(
        self,
        markup_type: MarkupTypes,
        description: str | Sequence[str],
        label: str | Sequence[str],
        position: tuple[float, float, float] | Sequence[tuple[float, float, float]],
        display: DisplaySettings | None = None,
        visible_points: bool = True,
    ) -> None:
        if display is None:
            display = DisplaySettings(
                color=(0.0, 1.0, 0.0),
                selected_color=(0.0, 1.0, 0.0),
                active_color=(0.0, 1.0, 0.0),
            )

        if isinstance(description, str):
            description = [description]
        if isinstance(label, str):
            label = [label]
        if isinstance(position[0], float):
            position = [position]

        markup = Markup(markup_type, display)
        for lab, desc, pos in zip(label, description, position, strict=True):
            markup.add_control_point(lab, desc, pos, visible_points)
        self.markups.append(markup)

    def add_fiducial(
        self,
        label: str | list[str],
        description: str | list[str],
        position: tuple[float, float, float] | list[tuple[float, float, float]],
        display: DisplaySettings | None = None,
        visible_points: bool = True,
    ) -> None:
        """
        Add a fiducial markup.

        Parameters
        ----------
        label : str or list[str]
            Label of the markup.
        description : str or list[str]
            Description of the markup.
        position : tuple[float, float, float] or list[tuple[float, float, float]]
            Position of the markup.
        display : DisplaySettings, optional
            Display settings of the markup.
        visible_points : bool
            Whether the points are visible or not.

        Returns
        -------
        None
        """
        self._add_item(MarkupTypes.FIDUCIAL, description, label, position, display, visible_points)

    def add_line(
        self,
        label: tuple[str, str],
        description: tuple[str, str],
        position: tuple[tuple[float, float, float], tuple[float, float, float]],
        display: DisplaySettings | None = None,
        visible_points: bool = True,
    ) -> None:
        """
        Add a line markup.

        Parameters
        ----------
        label : tuple[str, str]
            Label of the markup.
        description : tuple[str, str]
            Description of the markup.
        position : tuple[tuple[float, float, float], tuple[float, float, float]]
        display : DisplaySettings, optional
            Display settings of the markup.
        visible_points : bool
            Whether the points are visible or not.

        Returns
        -------
        None
        """
        self._add_item(MarkupTypes.LINE, description, label, position, display, visible_points)

    def add_angle(
        self,
        label: tuple[str, str, str],
        description: tuple[str, str, str],
        position: tuple[
            tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
        ],
        display: DisplaySettings | None = None,
        visible_points: bool = True,
    ) -> None:
        """
        Add an angle markup.

        Parameters
        ----------
        label : tuple[str, str, str]
            Label of the markup.
        description : tuple[str, str, str]
            Description of the markup.
        position : tuple[
                        tuple[float, float, float],
                        tuple[float, float, float],
                        tuple[float, float, float]
                    ]
            Position of the markup.
        display : DisplaySettings, optional
            Display settings of the markup.
        visible_points : bool
            Whether the points are visible or not.

        Returns
        -------
        None
        """
        self._add_item(MarkupTypes.ANGLE, description, label, position, display, visible_points)

    def add_curve(
        self,
        label: Sequence[str],
        description: Sequence[str],
        position: Sequence[tuple[float, float, float]],
        display: DisplaySettings | None = None,
        visible_points: bool = True,
    ) -> None:
        """
        Add a curve markup.

        Parameters
        ----------
        label : list[str]
            Label of the markup.
        description : list[str]
            Description of the markup.
        position : list[tuple[float, float, float]]
            Position of the markup.
        display : DisplaySettings, optional
            Display settings of the markup.
        visible_points : bool
            Whether the points are visible or not.

        Returns
        -------
        None
        """
        self._add_item(MarkupTypes.CURVE, description, label, position, display, visible_points)

    @staticmethod
    def load(path: Path) -> "MarkupManager":
        """
        Load a markup manager from a file.

        Parameters
        ----------
        path : Path
            Path to the file.

        Returns
        -------
        MarkupManager
            Markup manager loaded from the file.
        """
        with path.open("r") as f:
            data = json.load(f)
        return MarkupManager.from_dict(data)

    def save(self, path: Path) -> None:
        """
        Save the markup manager to a file.

        Parameters
        ----------
        path : Path
            Path to the file.

        Returns
        -------
        None
        """
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def from_dict(data: dict) -> "MarkupManager":
        """
        Load a markup manager from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing the data.

        Returns
        -------
        MarkupManager
            Markup manager loaded from the dictionary.
        """
        mm = MarkupManager()
        mm.markups = [Markup.from_dict(markup) for markup in data["markups"]]
        return mm

    def to_dict(self) -> dict:
        """
        Convert the markup manager to a dictionary.

        Returns
        -------
        dict
            Dictionary containing the data.
        """
        return {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules"
            "/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
            "markups": [m.to_dict() for m in self.markups],
        }
