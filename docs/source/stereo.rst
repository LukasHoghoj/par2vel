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

The partial 12 partial derivatives are set up in a matrix for each point, where stereo
PIV is performed, and the following overdefined equation is solved:

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

The ``lstsq`` mean square method algorythm from the ``numpy.linalg`` package is used
to solve these overdefined equations. The total displacements are returned in a 
:math:`3\times n_x\times n_y` 3D matrix, that is stored as ``dX`` in the ``Field3D``
object.