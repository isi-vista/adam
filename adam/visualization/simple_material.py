from panda3d.core import Material

"""Class for creating basic materials (i.e. color settings for geometry).
   Meant to insulate other code from any specific details about shaders."""


class SimpleMaterial:
    def __init__(
        self, r: int, g: int, b: int, alpha: int = 255, *, name: str = None
    ) -> None:
        if name is not None:
            self.mat = Material(name)
        else:
            self.mat = Material()

        self.mat.setDiffuse((r, g, b, alpha))  # default is fully opaque

    # could apply other global settings re: reflectiveness and other parameters here.
