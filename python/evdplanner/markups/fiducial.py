from dataclasses import dataclass


@dataclass
class Fiducial:
    markup_id: int
    label: str
    description: str
    position: list[float] | None = None
    visible: bool = True

    @staticmethod
    def from_dict(data: dict) -> "Fiducial":
        return Fiducial(
            markup_id=data["id"],
            label=data["label"],
            description=data["description"],
            position=data["position"],
            visible=data["visibility"],
        )

    def to_dict(self):
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
            "locked": True,
            "visibility": self.visible,
            "positionStatus": "defined" if self.position else "undefined",
        }
