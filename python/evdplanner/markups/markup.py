from dataclasses import dataclass, field
from enum import Enum

from evdplanner.markups.display_settings import DisplaySettings
from evdplanner.markups.fiducial import Fiducial


class MarkupTypes(Enum):
    ANGLE: str = "Angle"
    CURVE: str = "Curve"
    FIDUCIAL: str = "Fiducial"
    LINE: str = "Line"


@dataclass
class Markup:
    markup_type: MarkupTypes = MarkupTypes.FIDUCIAL
    display_settings: DisplaySettings = field(default_factory=DisplaySettings.default)
    control_points: list[Fiducial] = field(default_factory=list)

    def add_control_point(
        self, label: str, description: str, position: list[float], visible: bool = True
    ):
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
        return Markup(
            markup_type=MarkupTypes(data["type"]),
            display_settings=DisplaySettings.from_dict(data["display"]),
            control_points=[Fiducial.from_dict(cp) for cp in data["controlPoints"]],
        )

    def to_dict(self):
        return {
            "type": self.markup_type,
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
