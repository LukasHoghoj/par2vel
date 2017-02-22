""" 3D processing of PIV images tacken with the Sheimpflug method,

Lukas Hoghoj february 2017, inspirated by "piv2d" from Knud Erik Meyer"""

import numpy
import scipy

import numpy as np
def trans3D(field1,field2):
    """ field1 & field2: field 2D instances that contain the interogation field
                         and the camera coordinates and their respectfull 
                         displacements for respectively the first and second 
                         camera"""
    # Unwrapping
    cam1 = field1.camera
    cam2 = field2.camera
    theta1 = cam1.theta
    theta2 = cam2.theta
    x1 = field1.x
    x2 = field2.x
    dx1 = field1.dx
    dx2 = field2.dx
    # distance between lense and image center
    a1 = (cam1.M+1)*cam1.focal_length/cam1.M
    a2 = (cam2.M+1)*cam2.focal_length/cam2.M
    # transform centers of displacements to physical coordinates:
    dummy, ni , nj = x1.shape
    if x1.shape != x2.shape:
        raise NameError('image formats must be the same')
    X1 = np.zeros((2, ni, nj))
    X2 = np.zeros((2, ni, nj))
    dX1_flat = np.zeros((2, ni, nj))
    dX2_flat = np.zeros((2, ni, nj))
    for i in range(ni):
        for j in range(nj):
            X1[:,i,j] = cam1.x2X(x1[:,i,j])
            dX1_flat[:,i,j] = cam1.dx2dX(x1[:,i,j],dx1[:,i,j])
            X2[:,i,j] = cam2.x2X(x2[:,i,j])
            dX2_flat[:,i,j] = cam2.dx2dX(x2[:,i,j],dx2[:,i,j])
    
    
    

def phi(a,theta,l):
    p = np.atan(l*np.cos(theta))/(a-l*np*sin(theta))
    return p