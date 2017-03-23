from par2vel.field import Field3D
from par2vel.camera import Scheimpflug , Pinhole
from par2vel.pinhole_calibration import Calibrate_Pinhole
import numpy as np
import unittest

cam = Scheimpflug((512,512))
cam.set_calibration(np.pi/3,1/100) 
X = np.random.rand(3,20) - 0.5
X[0] = X[0] * 2
X[1] = X[1] * 2
X[2] = X[1] * 0.005
x = cam.X2x(X)
C = np.array([[0.06, 0 , 256] , [0 , 0.06 , 256]])
a , b , c = Calibrate_Pinhole(X , x , C)