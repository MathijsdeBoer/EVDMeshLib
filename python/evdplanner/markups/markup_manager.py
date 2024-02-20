import json
from pathlib import Path
from typing import Sequence

from .display_settings import DisplaySettings
from .markup import Markup, MarkupTypes


class MarkupManager:
    markups: list[Markup]

    def __init__(self):
        self.markups = []

    def _add_item(
        self,
        markup_type: MarkupTypes,
        description: str | Sequence[str],
        label: str | Sequence[str],
        position: tuple[float, float, float] | Sequence[tuple[float, float, float]],
        display: DisplaySettings | None = None,
        visible_points: bool = True,
    ):
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
        for lab, desc, pos in zip(label, description, position):
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
        self._add_item(MarkupTypes.FIDUCIAL, description, label, position, display, visible_points)

    def add_line(
        self,
        label: tuple[str, str],
        description: tuple[str, str],
        position: tuple[tuple[float, float, float], tuple[float, float, float]],
        display: DisplaySettings | None = None,
        visible_points: bool = True,
    ) -> None:
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
        self._add_item(MarkupTypes.ANGLE, description, label, position, display, visible_points)

    def add_curve(
        self,
        label: Sequence[str],
        description: Sequence[str],
        position: Sequence[tuple[float, float, float]],
        display: DisplaySettings | None = None,
        visible_points: bool = True,
    ) -> None:
        self._add_item(MarkupTypes.CURVE, description, label, position, display, visible_points)

    @staticmethod
    def from_file(path: Path) -> "MarkupManager":
        with path.open("r") as f:
            data = json.load(f)
        return MarkupManager.from_dict(data)

    @staticmethod
    def from_dict(data: dict) -> "MarkupManager":
        mm = MarkupManager()
        mm.markups = [Markup.from_dict(markup) for markup in data["markups"]]
        return mm

    def to_dict(self) -> dict:
        return {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules"
            "/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
            "markups": [m.to_dict() for m in self.markups],
        }
