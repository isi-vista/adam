import bpy # blender interface

#print(bpy.utils.modules_from_path("/Users/cjenkins/Documents/adam", []))

#from adam.geon import CrossSection

# use brew cask install blender to get nice command line access

# use: blender --background --python <script>

# It is necessary to specify a full path (no ~ shortcuts)
PATH = "/Users/cjenkins/Documents/ADAM_MISC/blender_test/"

# bpy.context.scene
cam = bpy.context.scene.camera
cam.location = (13, -9, 4.95)
    # https://docs.blender.org/api/current/bpy.types.Camera.html#bpy.types.Camera




def output(path: str, res_x: int, res_y: int) -> None:
    bpy.context.scene.render.filepath = path
    bpy.context.scene.render.resolution_x = res_x
    bpy.context.scene.render.resolution_y = res_y
    bpy.ops.render.render(use_viewport=True, write_still=True)

"""create ground plane"""
def make_plane():
    pass



# blender boilerplate
def register():
    pass
    #bpy.utils.register_module(adam)

def unregister():
    pass

if __name__ == "__main__":

    # delete default cube that is in default scene
    objs = bpy.data.objects
    objs.remove(objs["Cube"], do_unlink=True)

    bpy.ops.mesh.primitive_cube_add(location=(5.0, 1.0, 0.0))
    # output(PATH + "cube", 128, 128)

    # assign current focused object to a variable
    new_cube = bpy.context.object
    # type is bpy_types.Object
    print(type(new_cube))

    #bpy.ops.mesh.primitive_torus_add()

    bpy.ops.mesh.primitive_uv_sphere_add(location=(1.0, 1.0, 0.0))
    output(PATH + "cubeSphere", 256, 256)

    with open ("sample.txt") as f:
        for line in f:
            print(line)