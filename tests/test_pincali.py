from par2vel.field import Field3D
from par2vel.camera import Scheimpflug , Pinhole
from par2vel.pinhole_calibration import Calibrate_Pinhole
import numpy as np
import unittest


class Calibrationtest(unittest.TestCase):
    def testCalibration(self):
        cam = Scheimpflug((512,512))
        cam.set_calibration(np.pi/3,1/100) 
        X = np.random.rand(3,20) - 0.5
        X[0] = X[0] * 2
        X[1] = X[1] * 2
        X[2] = X[1] * 0.005
        x = cam.X2x(X)
        C = np.array([[0.06, 0 , 256] , [0 , 0.06 , 256]])
        R, dis ,C = Calibrate_Pinhole(X , x , C)
        pincam = Pinhole((512,512))
        pincam.manual_calibration(R, dis, C)
        X_test = np.array([[0.2, 1, -2, 0],[2, -1, 0, 0.1],[0.01, -0.04, -0.001, 0.02]])
        x_sch = cam.X2x(X_test)
        x_pin = pincam.X2x(X_test)
        numpy.testing.assert_array_almost_equal(x_pin,x_sch)

if __name__ == '__main__':
    unittest.main(exit = False)