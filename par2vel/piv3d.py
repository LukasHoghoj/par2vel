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

def stereo(field):
    """Uses the results from each camera plane, to find the 3D object plane
    displacements."""
    # Call function containing partial derivatives for movement at each point
    # in the fields
    field.dxdX()
    field.cam_dis()
    field.dX = np.zeros((field.size,3))
    """Probelms with nan values in camera displacements...
    Possible solutions: -mean of two linear interpolations in cam plane (one
    in x and one in y)
    -Let these points out and treat them as outliers
    -????
    """
    
    for i in range(field.size):
        field.dX[i,:] = np.linalg.lstsq(field.partial[i*4:(i+1)*4,i*3:(i+1)*3],field.dx_both[i*4:(i+1)*4])[0]
        