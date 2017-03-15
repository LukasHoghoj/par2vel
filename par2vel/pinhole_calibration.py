import numpy as np
from numpy.linalg import lstsq


def Calibrate_Pinhole(X, x):
    """Function to calibrate camera with respect to the pinhole model. The 
    function takes the input Calibrate_Pinhole([X,Y,Z],[x,y]); where X,Y and Z
    are the coordinates in object space and x and y are their respectevely 
    corresponding coordinates in the image plane"""



def Distortion(x_d, x_n):
    """Function returns the distortion cooeficients for given distorted and 
    normalized coordinates"""
    assert x_d.shape == x_n.shape
    len = x_d.shape[1]
    lhs = np.zeros(len * 2)
    i = np.indices(len * 2)
    lhs[i % 2 == 0] = x_d[0 , :] - x_n[0 , :]
    lhs[i % 2 == 1] = x_d[1 , :] - x_n[1 , :]
    rhs = np.zeros((len * 2,5))
    r = (x_n[0 , :] ** 2 + x_n[1 , :] ** 2)
    rhs[: , 0][i % 2 == 0] = x_n[0 , :] * r
    rhs[: , 0][i % 2 == 1] = x_n[1 , :] * r
    rhs[: , 1][i % 2 == 0] = x_n[0 , :] * r ** 2
    rhs[: , 1][i % 2 == 1] = x_n[1 , :] * r ** 2
    rhs[: , 2][i % 2 == 0] = x_n[0 , :] * r ** 3
    rhs[: , 2][i % 2 == 1] = x_n[1 , :] * r ** 3
    rhs[: , 3][i % 2 == 0] = 2 * x_n[0 , :] * x_n[1 , :]
    rhs[: , 3][i % 2 == 1] = r + 2 * x_n[1 , :] ** 2 
    rhs[: , 4][i % 2 == 0] = r + 2 * x_n[0 , :] ** 2
    rhs[: , 4][i % 2 == 1] = 2 * x_n[0 , :] * x_n[1 , :]
    
    dis = lstsq(rhs,lhs)[0]
    return dis

def Rotation_T()