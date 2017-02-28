""" 3D processing of PIV images tacken with the Sheimpflug method,

Lukas Hoghoj february 2017, inspirated by "piv2d" from Knud Erik Meyer"""

import numpy as np
import scipy

from par2vel.field import Field3D, Field2D
from par2vel.camera import Camera
from par2vel.piv2d import fftdx

def piv_camplane(Im_cam1,Im_cam2,field3d):
    """piv_camplane(Im_11,Im_12,Im_21,Im_22,field3d)
    Function that finds the displacements in both image planes and appends them
    to their field2d objects, that are contained in the global field3d"""
    fftdx(Im_cam1[0],Im_cam1[1],field3d.field2d[0])
    fftdx(Im_cam2[0],Im_cam2[1],field3d.field2d[1])

