# unittest of camera
# Copyright Knud Erik Meyer 2017
# Open source software under the terms of the GNU General Public License ver 3

import unittest
import numpy
import os
from par2vel.camera import *


class testCamera(unittest.TestCase):

    def test_X2x(self):
        cam = Camera((32,32))
        X = numpy.array([1.0, 1.0, 0]).reshape(3,-1)
        x = cam.X2x(X)
        self.assertAlmostEqual((x-[1, 1]).sum(),0)

    def test_x2X(self):
        cam = One2One((32,32))
        X = numpy.array([1.0, 1.0, 0]).reshape(3,-1)
        x = cam.X2x(X)
        X2 = cam.x2X(x)
        self.assertAlmostEqual((X-X2).sum(),0)

    def test_dx2dX(self):
        cam = One2One((32,32))
        dx = numpy.array([1.0, 1.0]).reshape(2,-1)
        x = dx
        dX = cam.dx2dX(x, dx)
        self.assertAlmostEqual((dX[0:2,:]-dx).sum(),0)

    def test_keywords(self):
        cam1 = Camera()
        cam1.focal_length = 0.06
        cam1.pixel_pitch = (0.00001, 0.00001)
        filename = 'temporary1.cam'
        cam1.save_camera(filename)
        cam2 = Camera()
        cam2.read_camera(filename)
        self.assertAlmostEqual(cam2.focal_length,0.06)
        self.assertAlmostEqual(cam2.pixel_pitch[1],0.00001)
        os.remove(filename)

##    def test_X2x(self):
##        cam = Camera((32,32))
##        cam.set_physical_size()
##        dx = numpy.array([[1],[0]])
##        dX = cam.dx2dX(dx,dx) 
##        self.assertAlmostEqual(numpy.sqrt((dX**2).sum()), 1)

class testLinear2d(unittest.TestCase):

    def test_X2x(self):
        from numpy import array
        cam = Linear2d()
        cam.set_calibration(array([[1, 0, 0.1], [0, 2, 0.2]]))
        X = array([[1, 0.5], [1, 0.6], [0.1, 0.2]])
        x_result = array([[1.1, 0.6],[2.2, 1.4]])
        x = cam.X2x(X)
        self.assertAlmostEqual((x - x_result).sum(),0)

    def test_x2X(self):
        from numpy import array
        cam = Linear2d()
        cam.set_calibration(array([[0.1, 0.01, 10], [0.02, 0.1, 11]]))
        X = array([[1, 0.5, -0.1], [1, 0.6, -0.12], [0, 0, 0]])
        x = cam.X2x(X)
        Xres = cam.x2X(x)
        self.assertAlmostEqual((X - Xres).sum(),0)        

    def test_dx2dX(self):
        from numpy import array
        cam = Linear2d()
        cam.set_calibration(array([[1, 0, 0.1], [0, 2, 0.2]]))
        x = array([[15.5, 32.5], [0, 15.5]])
        dx = array([[1, 1.2],[2,0]])
        dXres = array([[1, 1.2],[0, 1]])
        dX = cam.dx2dX(x,dx)
        self.assertAlmostEqual((dX - dXres).sum(), 0)

    def test_save_read(self):
        cam1 = Linear2d()
        cam1.focal_length = 0.06
        calib = numpy.array([[0.1, 0, 100], [0, 0.11, 101]])
        cam1.set_calibration(calib)
        filename = 'temporary2.cam'
        cam1.save_camera(filename)
        cam2 = Linear2d()
        cam2.read_camera(filename)
        self.assertAlmostEqual(cam2.focal_length,0.06)
        self.assertAlmostEqual((cam2.calib - calib).sum(), 0)
        os.remove(filename)

class testLinear3d(unittest.TestCase):

    def test_X2x(self):
        from numpy import array
        cam = Linear3d()
        matrix = array([[1.0, 0, 0, 0],
                        [0.0, 1, 0, 0],
                        [0.0, 0, 0, 1]])
        cam.set_calibration(matrix)
        X = array([[1.0], [1.0], [0]])
        x_result = array([[1.0],[1.0]])
        x = cam.X2x(X)
        self.assertAlmostEqual((x - x_result).sum(),0)

    def test_save_read(self):
        from numpy import array
        cam1 = Linear3d()
        cam1.focal_length = 0.04
        calib = array([[1.0, 0, 0, 0],
                       [0.0, 1, 0, 0],
                       [0.0, 0, 0, 1]])
        cam1.set_calibration(calib)
        filename = 'temporary4.cam'
        cam1.save_camera(filename)
        cam2 = Linear3d()
        cam2.read_camera(filename)
        self.assertAlmostEqual(cam2.focal_length,0.04)
        self.assertAlmostEqual((cam2.calib - calib).sum(), 0)
        os.remove(filename)


class testScheimpflug(unittest.TestCase):

    def test_X2x(self):
        from numpy import array, pi, sqrt
        cam = Scheimpflug((16,16))
        cam.focal_length = 0.06
        cam.pixel_pitch = (1e-5, 1e-5)
        cam.set_calibration(pi/4, 0.1)
        X = array([[0.0], [0.0], [0.0]])
        x_result = array([[7.5],[7.5]])
        x = cam.X2x(X)
        self.assertAlmostEqual((x - x_result).sum(),0)
        X = array([[-0.001], [0.0], [0.001]])
        x_result = array([[7.5],[7.5]])
        x = cam.X2x(X)
        self.assertAlmostEqual((x - x_result).sum(),0)
        X = array([[0.0001*sqrt(2)], [0.0], [0.0]])
        x_result = array([[8.5],[7.5]])
        x = cam.X2x(X)
        self.assertTrue(abs((x - x_result).sum())<0.005,0)

    def test_x2X(self):
        from numpy import array, pi
        cam = Scheimpflug()
        cam.set_calibration(pi/4, 0.1)
        X = array([[0.0, 0.01, 0.00,  0.01],
                   [0.0, 0.00, 0.01, -0.01],
                   [0,   0,    0,     0     ]])
        x = cam.X2x(X)
        Xresult = cam.x2X(x)
        self.assertAlmostEqual(abs((X - Xresult).sum()),0)
        
    def test_dx2dX(self):
        from numpy import array, pi, zeros
        cam = Scheimpflug()
        cam.set_calibration(pi/4, 0.1)
        X = array([[0.0], [0.0], [0.0]])
        dX = array([[0.001], [-0.002], [0.0]])
        x = cam.X2x(X)
        dx = cam.dX2dx(X, dX)
        dX2 = cam.dx2dX(x, dx)
        self.assertAlmostEqual(abs((dX - dX2).sum()),0)

    def test_save_read(self):
        cam1 = Scheimpflug()
        cam1.focal_length = 0.05
        cam1.set_calibration(numpy.pi/4, 0.1)
        filename = 'temporary3.cam'
        cam1.save_camera(filename)
        cam2 = Scheimpflug()
        cam2.read_camera(filename)
        self.assertAlmostEqual(cam2.focal_length,0.05)
        self.assertAlmostEqual(cam2.theta, numpy.pi/4)
        self.assertAlmostEqual(cam2.M, 0.1)
        os.remove(filename)

class testPinhole(unittest.TestCase):
    def test_X2x(self):
        import numpy as np
        cam  = Pinhole((512,512))
        R = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11,12]])
        dis = np.array([ 1e-2, 2e-2, 3e-2, 4e-2, 5e-2])
        C = np.array([[6e-2, 0, 256],
                      [0, 6e-2, 256]])
        cam.set_calibration(R, dis, C)
        X = np.array([[0], [0], [0]])
        x_result = np.array([[256.0237374], [256.0454749]])
        x = cam.X2x(X)
        self.assertAlmostEqual((x - x_result).sum(), 0)
        X = np.array([[1], [1], [1]])
        x_result = np.array([[256.0168076817485], [256.041324462342]])
        x = cam.X2x(X)
        self.assertAlmostEqual((x - x_result).sum(), 0)

    def testCalibration(self):
        """This test is primarily testing the calibration for the 
        pinhole model, however, it is also testing some functions of
        the Pinhole camera model as they are necessary for this unittest
        """
        import numpy as np
        cam = Scheimpflug((512,512))
        cam.set_calibration(np.pi/3,1/100)
        X = np.mgrid[-2:2:0.1,-2:2:0.1,-0.001:0.001:0.001]
        X = X.reshape((3,-1))
        x = cam.X2x(X)
        C = np.array([[0.06, 0 , 256] , [0 , 0.06 , 256]])
        pincam = Pinhole((512,512))
        pincam.calibration(x, X)
        X_test = np.array([[2, 1.1, -0.1, 1.5],[0.2, -1.1, 0, 0.4],[0.0005, -0.0003, -0.0009, 0.0009]])
        x_sch = cam.X2x(X_test)
        x_pin = pincam.X2x(X_test)
        zero = np.zeros(4)
        diff = np.sqrt((x_sch[0] - x_pin[0]) ** 2 + (x_sch[1] - x_pin[1]) ** 2)
        #numpy.testing.assert_array_almost_equal(diff,zero,decimal = 10)
        np.testing.assert_array_almost_equal(x_sch,x_pin,decimal = 8)
    
    def test_save_read(self):
        import numpy as np
        camw  = Pinhole((512,512))
        R = np.array([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 1.10, 1.0]])
        dis = np.array([ 1.0e-2, 2.0e-2, 3.0e-2, 4.0e-2, 5.0e-2])
        C = np.array([[6e-2, 0, 256.0],
                      [0, 6e-2, 256.0]])
        camw.set_calibration(R, dis, C)
        filename = 'temporary5.cam'
        camw.save_camera(filename)
        camr = Pinhole()
        camr.read_camera(filename)
        self.assertAlmostEqual((camw.R - camr.R).sum(),0)
        os.remove(filename)

    def test_x2X(self):
        import numpy as np
        # Setup a realistic camera based on Scheimpflug
        cam = Scheimpflug((512,512))
        cam.set_calibration(np.pi/3,1/100)
        X = np.mgrid[-2:2:0.1,-2:2:0.1,-0.001:0.001:0.001]
        X = X.reshape((3,-1))
        x = cam.X2x(X)
        C = np.array([[0.06, 0 , 256] , [0 , 0.06 , 256]])
        pincam = Pinhole((512,512))
        pincam.calibration(x, X)
        """
        # Bigger distortion
        pincam.k1 = -1.0e-7
        pincam.k2 = -2.0e-7
        pincam.k3 = -3.0e-7
        pincam.p1 = -4.0e-8
        pincam.p2 = -5.0e-8
        """
        X = np.array([[0,1.1,0.5,1], [0,1,-1,-2], [0,0,0,0]])
        x = pincam.X2x(X)
        X_computed = pincam.x2X(x)
        self.assertAlmostEqual(abs(X - X_computed).sum(),0)
        
      

if __name__=='__main__':
    numpy.set_printoptions(precision=4)
    unittest.main(exit = False)
