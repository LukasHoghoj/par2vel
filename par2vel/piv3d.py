""" 3D processing of PIV images tacken with the Sheimpflug method,

Lukas Hoghoj february 2017, inspirated by "piv2d" from Knud Erik Meyer"""

import numpy as np
import scipy

from par2vel.field import Field3D, Field2D
from par2vel.camera import Camera
from par2vel.piv2d import fftdx
import time

def piv_camplane(Im_cam1,Im_cam2,field3d):
    """piv_camplane(Im_11,Im_12,Im_21,Im_22,field3d)
    Function that finds the displacements in both image planes and appends them
    to their field2d objects, that are contained in the global field3d"""
    fftdx(Im_cam1[0],Im_cam1[1],field3d.field2d[0])
    fftdx(Im_cam2[0],Im_cam2[1],field3d.field2d[1])

def stereo(field):
    """Uses the results from each camera plane, to find the 3D object plane
    displacements."""
    from numpy.linalg import lstsq
    from scipy import sparse
    from scipy.sparse.linalg import lsqr
    from numpy import zeros, mgrid
    # Call function containing partial derivatives for movement at each point
    # in the fields
    field.dxdX()
    """Probelms with nan values in camera displacements...
    Possible solutions: -mean of two linear interpolations in cam plane (one
    in x and one in y)
    -Let these points out and treat them as outliers
    -????
    """
    part = sparse.lil_matrix((4 * field.size , 3 * field.size))
    j,k = mgrid[0 : 4 * field.size , 0 : 3 * field.size] 
    for i in range(len(field.field2d)):
        part[3*(j-2*i) == 4*k] = field.partial[i,0,:,:,:].reshape((field.X.shape[0]-1,-1))[0]
        part[3*(j-1-2*i) == 4*k] = field.partial[i,0,:,:,:].reshape((field.X.shape[0]-1,-1))[1]
        part[3*(j-2*i) == 4*(k-1)] = field.partial[i,1,:,:,:].reshape((field.X.shape[0]-1,-1))[0]
        part[3*(j-1-2*i) == 4*(k-1)] = field.partial[i,1,:,:,:].reshape((field.X.shape[0]-1,-1))[1]
        part[3*(j-2*i) == 4*(k-2)] = field.partial[i,2,:,:,:].reshape((field.X.shape[0]-1,-1))[0]
        part[3*(j-1-2*i) == 4*(k-2)] = field.partial[i,2,:,:,:].reshape((field.X.shape[0]-1,-1))[1]
    field.cam_dis()
    t = time.time()
    dX1 = lsqr(part,field.dx_both)[0]
    print("Time to solve the equation %s seconds" % (time.time()-t))
    dX1 = dX1.reshape((field.size,3))
    field.dX = np.zeros((3,field.res[1],field.res[0]))
    field.dX[0,:,:] = dX1[:,0].reshape((field.res[1],field.res[0]))
    field.dX[1,:,:] = dX1[:,1].reshape((field.res[1],field.res[0]))
    field.dX[2,:,:] = dX1[:,2].reshape((field.res[1],field.res[0]))