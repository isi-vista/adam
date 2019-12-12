""" Visualization module: Code for producing more human-friendly representations of scenes from curricula.

    Main executable:
        make_scenes.py:
            Converts perceptual representations of scenes into 3d rendered scenes.

            Requires a parameters file to be run.

    Other modules:
        panda3d_interface.py:
            Interface for the Panda3D rendering engine. Responsible for creating, destroying, positioning, coloring, etc
            objects that are arranged in 3D space to represent perceptions of scenes.

        positioning.py:
            Module defining a Stochastic Gradient Descent model for arranging objects in a scene. Responsible for
            translating relative layout requirements into a concrete layout.

        utils.py:
            Miscellaneous type definitions used by multiple visualization modules.

"""
