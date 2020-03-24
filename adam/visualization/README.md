Example for running rendering engine over the scenes in an experiment:

``python -m adam.visualization.render_curriculum parameters/visualization/m9_debug.params
``

If you would like to link the HTML output from an experiment to images produced from visualization, ensure that the
experiment's params file contains: ``include_image_links: True`` 


If you use ``parameters/visualization/render_curriculum.params`` instead, it will render all phase1 scenes that can currently be handled. (Separate folders for each curriculum)

