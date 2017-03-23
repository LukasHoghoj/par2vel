import numpy as np
from scipy.linalg import lstsq
import scipy as sp

def Calibrate_Pinhole(X, x, C):
    """Function to calibrate camera with respect to the pinhole model. The
    function takes the input Calibrate_Pinhole([X,Y,Z],[x,y]); where X,Y and Z
    are the coordinates in object space and x and y are their respectevely
    corresponding coordinates in the image plane"""
    assert X.shape[1] == x.shape[1]
    # Maximum number of iterations:
    ite_max = 1000
    # Maximum error:
    err_max = 0.5
    len = X.shape[1]
    X_p = np.vstack((X, np.ones(len)))
    x_d = np.zeros(x.shape)
    x_d[0, :] = (x[0, :] - C[0, 2]) / C[0, 0]
    x_d[1, :] = (x[1, :] - C[1, 2]) / C[1, 1]
    R = Rotation_T(x_d, X)
    X_C = np.dot(R, X_p)
    x_n = np.zeros((2, len))
    x_n[0] = X_C[0]/X_C[2]
    x_n[1] = X_C[1] / X_C[2]
    k1, k2, k3, p1, p2 = Distortion(x_d, x_n)
    r = x_n[0, :] ** 2 + x_n[1, :] ** 2
    x_d[0, :] = x_n[0, :] * (1 + k1 * r + k2 * r ** 2 + k3 * r ** 3) + 2 * \
                p1 * x_n[0, :] * x_n[1, :] + p2 * (r + 2 * x_n[0, :] ** 2)
    x_d[1, :] = x_n[1, :] * (1 + k1 * r + k2 * r ** 2 + k3 * r ** 3) + p1 *\
                 (r + 2 * x_n[1, :] ** 2) + 2 * p2 * x_n[0, :] * x_n[1, :]
    C = Cam_Matrix(x, x_d)
    x_p = np.dot(C, np.vstack((x_d, np.ones(len))))
    # Compute average error:
    err = np.mean(np.sqrt((x_p[0] - x[0]) ** 2 + (x_p[1] - x[1]) ** 2))
    print(err)
    """
    ite = 0
    while err >= err_max or ite < ite_max:
        # Recompute through the system to find the error:
        X_C = np.dot(R,X_p)
        x_n = np.zeros((len, 2))
        x_n[0] = X_C[0]/X_C[2]
        x_n[1] = X_C[1] / X_C[2]
        r = x_n[0, :] ** 2 + x_n[1, :] ** 2
        x_d[0, :] = x_n[0, :] * (1 + k1 * r + k2 * r ** 2 + k3 * r ** 3) + 2 * \
                p1 * x_n[0, :] * x_n[1, :] + p2 * (r + 2 * x_n[0, :] ** 2)
        x_d[1, :] = x_n[1, :] * (1 + k1 * r + k2 * r ** 2 + k3 * r ** 3) + p1 *\
                 (r + 2 * x_n[1, :] ** 2) + 2 * p2 * x_n[0, :] * x_n[1, :]
        x_p = np.dot(C, x_d)
        err = np.mean(np.sqrt((x_p[0] - x[0]) ** 2 + (x_p[1] - x[1]) ** 2))
    """ 
    print(err)
    dis = np.array([k1, k2, k3, p1, p2])
    return R, dis, C

def Distortion(x_d, x_n):
    """Function returns the distortion cooeficients for given distorted and 
    normalized coordinates"""
    assert x_d.shape == x_n.shape
    len = x_d.shape[1]
    lhs = np.zeros(len * 2)
    i = np.arange(len * 2)
    lhs[i % 2 == 0] = x_d[0, :] - x_n[0, :]
    lhs[i % 2 == 1] = x_d[1, :] - x_n[1, :]
    rhs = np.zeros((len * 2, 5))
    r = (x_n[0, :] ** 2 + x_n[1, :] ** 2)
    rhs[:, 0][i % 2 == 0] = x_n[0, :] * r
    rhs[:, 0][i % 2 == 1] = x_n[1, :] * r
    rhs[:, 1][i % 2 == 0] = x_n[0, :] * r ** 2
    rhs[:, 1][i % 2 == 1] = x_n[1, :] * r ** 2
    rhs[:, 2][i % 2 == 0] = x_n[0, :] * r ** 3
    rhs[:, 2][i % 2 == 1] = x_n[1, :] * r ** 3
    rhs[:, 3][i % 2 == 0] = 2 * x_n[0, :] * x_n[1, :]
    rhs[:, 3][i % 2 == 1] = r + 2 * x_n[1, :] ** 2 
    rhs[:, 4][i % 2 == 0] = r + 2 * x_n[0, :] ** 2
    rhs[:, 4][i % 2 == 1] = 2 * x_n[0, :] * x_n[1, :]
    
    dis = lstsq(rhs,lhs)[0]
    return dis

def Rotation_T(x_n, X_p):
    """Function that finds the rotation and translation parameters """
    assert x_n.shape[1] == X_p.shape[1]
    len = x_n.shape[1]
    lhs = np.zeros(len * 2)
    rhs = np.zeros((len * 2, 11))
    i = np.arange(len * 2)
    lhs[i % 2 == 0] = x_n[0, :]
    lhs[i % 2 == 1] = x_n[1, :]
    rhs[:, 0 : 3][i % 2 == 0] = X_p.T
    rhs[:, 3][i % 2 == 0] = 1
    rhs[:, 0 : 4][i % 2 == 1] = 0
    rhs[:, 4 : 8][i % 2 == 0] = 0
    rhs[:, 4 : 7][i % 2 == 1] = X_p.T
    rhs[:, 7][i % 2 == 1] = 1
    rhs[:, 8][i % 2 == 0] = - x_n[0, :] * X_p[0, :]
    rhs[:, 8][i % 2 == 1] = - x_n[1, :] * X_p[0, :]
    rhs[:, 9][i % 2 == 0] = - x_n[0, :] * X_p[1, :]
    rhs[:, 9][i % 2 == 1] = - x_n[1, :] * X_p[1, :]
    rhs[:, 10][i % 2 == 0] = - x_n[0, :] * X_p[2, :]
    rhs[:, 10][i % 2 == 1] = - x_n[1, :] * X_p[2, :]
    #rhs[:, 11][i % 2 == 0] = - x_n[0, :]
    #rhs[:, 11][i % 2 == 1] = - x_n[1, :]
    RT = lstsq(rhs,lhs)[0]
    R = np.zeros((3,4))
    R[0] = RT[0 : 4]
    R[1] = RT[4 : 8]
    R[2, 0 : 3] = RT[8 : 12]
    R[2, 3] = 1
    return R

def Cam_Matrix(x_p, x_d):
    """Function that optimizes the camera matrix, such that it fits the best
    """
    assert x_p.shape == x_d.shape
    len = x_p.shape[1]
    lhs = np.zeros(len * 2)
    i = np.arange(len * 2)
    rhs = np.zeros((len * 2, 4))
    lhs[i % 2 == 0] = x_p[0, :]
    lhs[i % 2 == 1] = x_p[1, :]
    rhs[:, 0][i % 2 == 0] = x_d[0, :]
    rhs[:, 0][i % 2 == 1] = 0
    rhs[:, 1][i % 2 == 0] = 0
    rhs[:, 1][i % 2 == 1] = x_d[1, :]
    rhs[:, 2][i % 2 == 0] = 1
    rhs[:, 2][i % 2 == 1] = 0
    rhs[:, 3][i % 2 == 0] = 0
    rhs[:, 3][i % 2 == 1] = 1

    f = lstsq(rhs, lhs)[0]
    Cam_M = np.zeros((2,3))
    Cam_M[0, 0] = f[0]
    Cam_M[1, 1] = f[1]
    Cam_M[0, 2] = f[2]
    Cam_M[1, 2] = f[3]

    return Cam_M