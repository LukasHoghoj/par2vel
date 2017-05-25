===============
Stereo-PIV
===============

While classical PIV only permits it to determinate the particle displacements in
two dimensions; stereoscopic PIV uses the displacements in two camera planes in
order to find the total, 3 dimensional, displacements in the object plane. The 
function ``piv_camplane``, which takes the inputs listed below, will call the 
necessary functions to find the camera plane displacements:

1. An array with both images for the first camera ``[Image_1, Image_2]``
2. An array with both images for the second camera ``[Image_1, Image_2]``
3. The already initiated ``Field3D`` object, where the grid function already has
   been called.

To perform stereo-PIV, the :ref:`Field3D` functions are used to set up a grid in
both camera planes. The images are then used to find both camera plane 
displacements. As the ``Field3D`` object has been initialized and the grid already
has been created, the ``stereo`` function can be called; the function takes the 
``Field3D`` object as input. The field's ``dxdX()`` function will first be called in
order to compute the :ref:`Partial derivatives`. 

^^^^^^^^^^^^^^^
Data structure
^^^^^^^^^^^^^^^

In order to solve for the 3D displacement vectors in the object plane, the following
overdefined equation has to be solved for each point in the grid:

.. math::

    \left[\begin{array}{c}\Delta x_1\\\Delta y_1\\\Delta x_2\\\Delta y_2\end{array}
    \right] = \left[\begin{array}{ccc}
    \frac{\partial x_1}{\partial X}&\frac{\partial x_1}{\partial Y}&
    \frac{\partial x_1}{\partial Z}\\
    \frac{\partial y_1}{\partial X}&\frac{\partial y_1}{\partial Y}&
    \frac{\partial y_1}{\partial Z}\\
    \frac{\partial x_2}{\partial X}&\frac{\partial x_2}{\partial Y}&
    \frac{\partial x_2}{\partial Z}\\
    \frac{\partial y_2}{\partial X}&\frac{\partial y_2}{\partial Y}&
    \frac{\partial y_2}{\partial Z}
    \end{array}\right]\cdot\left[\begin{array}{c}\Delta X\\\Delta Y\\\Delta Z
    \end{array}\right]

The :math:`4\times3` matrix in the equation above, just as the left hand side vector,
are set up for each point in the grid. The ``numpy.linalg`` function ``lstsq`` is used
to solve the above equation seperately for each point in the grid. The results are saved
in the displacement array ``dX`` in the ``Field3D`` object, which takes the form of a 
:math:`3\times n_y\times n_x` 3D-matrix.