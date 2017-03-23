from par2vel.field import Field3D
from par2vel.camera import Scheimpflug , Pinhole
from par2vel.pinhole_calibration import Calibrate_Pinhole
import numpy as np
import unittest


class Calibrationtest(unittest.TestCase):
    def testCalibration(self):
        """This test is primarily testing the calibration for the 
        pinhole model, however, it is also testing some functions of
        the Pinhole camera model as they are necessary for this unittest
        """
        cam = Scheimpflug((512,512))
        cam.set_calibration(np.pi/3,1/100)
        X = np.mgrid[-2:2:0.05,-2:2:0.05,-0.001:0.001:0.0005]
        print(X.shape)
        X = X.reshape((3,-1))
        x = cam.X2x(X)
        C = np.array([[0.06, 0 , 256] , [0 , 0.06 , 256]])
        R, dis ,C = Calibrate_Pinhole(X , x , C)
        pincam = Pinhole((512,512))
        pincam.manual_calibration(R, dis, C)
        X_test = np.array([[2, 1.1, -0.1, 1.5],[0.2, -1.1, 0, 0.4],[0.0005, -0.0003, -0.0009, 0.0009]])
        x_sch = cam.X2x(X_test)
        x_pin = pincam.X2x(X_test)
        zero = np.zeros(4)
        diff = np.sqrt((x_sch[0] - x_pin[0]) ** 2 + (x_sch[1] - x_pin[1]) ** 2)
        #numpy.testing.assert_array_almost_equal(diff,zero,decimal = 10)
        numpy.testing.assert_array_almost_equal(x_sch,x_pin,decimal = 8)

if __name__ == '__main__':
    unittest.main(exit = False)