"""
The DisplaySettings class, which is used to store the display settings for a 3D Slicer model node.
"""

from dataclasses import dataclass, field


@dataclass
class DisplaySettings:
    """
    Class to store the display settings for a 3D Slicer model node.
    """

    color: tuple[float, float, float]
    selected_color: tuple[float, float, float]
    active_color: tuple[float, float, float]

    visibility: bool = True
    opacity: float = 1.0
    properties_label_visibility: bool = False
    point_labels_visibility: bool = True

    text_scale: float = 3.0

    glyph_type: str = "Sphere3D"
    glyph_scale: float = 3.0
    glyph_size: float = 5.0

    slice_projection: bool = False
    slice_projection_use_fiducial_color: bool = True
    slice_projection_outlined_behind_slice_plane: bool = False
    slice_projection_color: tuple[float, float, float] = field(default_factory=tuple)
    slice_projection_opacity: float = 2.0

    line_thickness: float = 0.2
    line_color_fading_start: float = 1.0
    line_color_fading_end: float = 10.0
    line_color_fading_hue_offset: bool = 0.0
    line_color_fading_saturation: bool = 1.0

    handles_interactive: bool = False
    translation_handle_visibility: bool = False
    rotation_handle_visibility: bool = False
    scale_handle_visibility: bool = False
    interaction_handle_scale: float = 3.0

    snap_mode: str = "toVisibleSurface"

    def __init__(
        self,
        color: tuple[float, float, float],
        selected_color: tuple[float, float, float],
        active_color: tuple[float, float, float],
    ) -> None:
        """
        Initialize the DisplaySettings object.

        Parameters
        ----------
        color : tuple[float, float, float]
            The color of the markups.
        selected_color : tuple[float, float, float]
            The color of the markups when selected.
        active_color : tuple[float, float, float]
            The color of the markups when active.
        """
        super().__init__()
        self.color = color
        self.selected_color = selected_color
        self.active_color = active_color

        self.slice_projection_color = self.color

    @classmethod
    def default(cls: type["DisplaySettings"]) -> "DisplaySettings":
        """
        Create a DisplaySettings object with default settings.

        Returns
        -------
        DisplaySettings
            The default display settings.
        """
        return cls(
            color=(1, 0, 0),
            selected_color=(0, 1, 0),
            active_color=(0, 1, 0),
        )

    @staticmethod
    def from_dict(data: dict) -> "DisplaySettings":
        """
        Create a DisplaySettings object from a dictionary.

        Parameters
        ----------
        data : dict
            The dictionary containing the display settings.

        Returns
        -------
        DisplaySettings
            The display settings.
        """
        settings = DisplaySettings(
            color=data["color"],
            selected_color=data["selectedColor"],
            active_color=data["activeColor"],
        )
        settings.visibility = data["visibility"]
        settings.opacity = data["opacity"]
        settings.properties_label_visibility = data["propertiesLabelVisibility"]
        settings.point_labels_visibility = data["pointLabelsVisibility"]
        settings.text_scale = data["textScale"]
        settings.glyph_type = data["glyphType"]
        settings.glyph_scale = data["glyphScale"]
        settings.glyph_size = data["glyphSize"]
        settings.slice_projection = data["sliceProjection"]
        settings.slice_projection_use_fiducial_color = data["sliceProjectionUseFiducialColor"]
        settings.slice_projection_outlined_behind_slice_plane = data[
            "sliceProjectionOutlinedBehindSlicePlane"
        ]
        settings.slice_projection_color = data["sliceProjectionColor"]
        settings.slice_projection_opacity = data["sliceProjectionOpacity"]
        settings.line_thickness = data["lineThickness"]
        settings.line_color_fading_start = data["lineColorFadingStart"]
        settings.line_color_fading_end = data["lineColorFadingEnd"]
        settings.line_color_fading_hue_offset = data["lineColorFadingHueOffset"]
        settings.line_color_fading_saturation = data["lineColorFadingSaturation"]
        settings.handles_interactive = data["handlesInteractive"]
        settings.translation_handle_visibility = data["translationHandleVisibility"]
        settings.rotation_handle_visibility = data["rotationHandleVisibility"]
        settings.scale_handle_visibility = data["scaleHandleVisibility"]
        settings.interaction_handle_scale = data["interactionHandleScale"]
        settings.snap_mode = data["snapMode"]
        return settings

    def to_dict(self) -> dict:
        """
        Convert the DisplaySettings object to a dictionary.

        Returns
        -------
        dict
            The dictionary containing the display settings.
        """
        return {
            "visibility": self.visibility,
            "opacity": self.opacity,
            "color": self.color,
            "selectedColor": self.selected_color,
            "activeColor": self.active_color,
            "propertiesLabelVisibility": self.properties_label_visibility,
            "pointLabelsVisibility": self.point_labels_visibility,
            "textScale": self.text_scale,
            "glyphType": self.glyph_type,
            "glyphScale": self.glyph_scale,
            "glyphSize": self.glyph_size,
            "sliceProjection": self.slice_projection,
            "sliceProjectionUseFiducialColor": self.slice_projection_use_fiducial_color,
            "sliceProjectionOutlinedBehindSlicePlane": self.slice_projection_outlined_behind_slice,
            "sliceProjectionColor": self.slice_projection_color,
            "sliceProjectionOpacity": self.slice_projection_opacity,
            "lineThickness": self.line_thickness,
            "lineColorFadingStart": self.line_color_fading_start,
            "lineColorFadingEnd": self.line_color_fading_end,
            "lineColorFadingHueOffset": self.line_color_fading_hue_offset,
            "lineColorFadingSaturation": self.line_color_fading_saturation,
            "handlesInteractive": self.handles_interactive,
            "translationHandleVisibility": self.translation_handle_visibility,
            "rotationHandleVisibility": self.rotation_handle_visibility,
            "scaleHandleVisibility": self.scale_handle_visibility,
            "interactionHandleScale": self.interaction_handle_scale,
            "snapMode": self.snap_mode,
        }
