==============
Field
==============

The result of an analysis is a velocity field. This is stored and manupilated
in the field class. This class actually operates on displacements in pixels.
This is the raw data from the image analysis. The class will also be able
to return the field in as velocities in m/s. However, this is not implemented
in the current version at the moment. The current version only works 
with inplane vectores in planar fields, also known as 2C2D fields. This is 
handled with the Field2D class. New classes will be developed later to
hold planer fields with three velocity components (stereoscopic PIV) and 
volumetric fields.

---------------------
Field2D
---------------------
The internal data in this objects consist of several elements:

Camera:
    A field has a camera and thereby know the relation between physical
    (object) koordinates and image coordinates. 
Coordinates:
    The interrogation grid is stored in the object together with 
    interrogation areas (IA). 
Outlier detection:
    The object contains functions used for outlier detection and replacement. 
    It also keeps track of which vectors that were replaced.

---------------------
Field3D
---------------------
Field3D object contains multiple ``Field2D`` objects, one for each used camera. The aim 
of this function is to set up all the data necessary in order to perform stereo PIV.

^^^^^^^^^^^^^^^^^^^^^
Working area
^^^^^^^^^^^^^^^^^^^^^

The working area, is considered as the area, in object plane, that both cameras cover.
The four corners of each camera are given by their sizes and are transformed to the 
object plane using the camera models. The biggest rectangle, where there still is data
everywhere from both cameras, is now used as the working area, i.e. the area where the
stereo PIV analysis will be performed.

^^^^^^^^^^^^^^^^^^^^^
Setup of a grid
^^^^^^^^^^^^^^^^^^^^^

In order to obtain a consistant grid, where the displacements in both camera planes 
can be set to one universal grid in the physical space, the grid is first created in 
object plane. Once the ``Field3D.grid([nx,ny])`` function has been called, a grid 
:math:`nX` wide and :math:`nY` high will be created in the object plane (all :math:`Z`
components are set to zero). All points in the grid are equally spaced.

The grid in object space is transformed to each camera plane, in order to determine the
camera plane displacements.

^^^^^^^^^^^^^^^^^^^^^
Partial derivatives
^^^^^^^^^^^^^^^^^^^^^

In order to become able to determine the displacements in object plane with any camera model
(that eventually is very computationaly expensive to transform from camera to object plane),
the displacement partial derivatives are found at each interrogation point. This is done by:

.. math::

   \frac{\partial\mathbf{x^\intercal}}{\partial\mathbf{X}} = dX2dx\left(\left[\begin{array}{ccc}
                    1   &   0   &   0\\
                    0   &   1   &   0\\
                    0   &   0   &   1
   \end{array}\right]\times10^{-3}\right)\times10^3

The results of these operations are stored in a 5D array, that takes the following entries in their
respective order:

0. The index of the camera in use (corresponding to the cameras index in the ``field2d```
   subobject in ``Field3D``)
1. The coordinate in object space, that has been used
2. The coordinate in camera plane, that has been derived
3. The :math:`nX` index that can be related to the :math:`X` coordinate, when looking at the grid
4. The :math:`nY` index that can be related to the :math:`Y` coordinate, when looking at the grid