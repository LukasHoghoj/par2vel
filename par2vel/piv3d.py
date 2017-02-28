""" 3D processing of PIV images tacken with the Sheimpflug method,

Lukas Hoghoj february 2017, inspirated by "piv2d" from Knud Erik Meyer"""

import numpy as np
import scipy

from par2vel.field import Field3D, Field2D
from par2vel.Camera import Camera
from par2vel.piv2d import fftdx

def piv_camplane(Im_11,Im_12,Im_21,Im_22,field3d):
    """piv_camplane(Im_11,Im_12,Im_21,Im_22,field3d)
    Function that finds the displacements in both image planes and appends them
    to their field2d objects, that are contained in the global field3d"""
    fftdx(Im_11,Im_12,field3d.field2d[0])
    fftdx(Im_21,Im_22,field3d.field2d[1])

