

USAGE_MESSAGE = """Takes a list of geometry with properties: location, orientation, color, scale
 and interfaces with blender to produce an output render of them."""

# use brew cask install blender to get nice command line access

# use: blender --background --python <script>

import bpy


from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point


# can possibly have a wide angle lens for initial testing purposes: just create objects,
# worry about positioning them later

def main():
    pass

if __name__ == "__main__":
    parameters_only_entry_point(main, usage_message=USAGE_MESSAGE)