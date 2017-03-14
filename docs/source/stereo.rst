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

A :math:`(4\cdot n_x\cdot n_y)\times(3\cdot n_x\cdot n_y)` sparse matrix is created, that
contains the :math:`4\times3` matrices from the equation above in it's diagonal (one for
each interogation point). All vectors containing the displacements in the camera
planes (see left hand side from the equation above) are also stacked together; in the
same order as the system matrices. As the sparse matrix has been created with the
``scipy.sparse.lil_matrix`` command, the ``scipy.sparse.linalg.lsqr`` least square
solver has to be used (the first array of the returned data is the solution to
the equation). The data is then rearanged in such way that it the displacement
array ``dX`` in the ``Field3D`` object takes the form of a 
:math:`3\times n_y\times n_x` 3D-matrix.