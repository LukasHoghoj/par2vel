#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:26:50 2017

@author: Lukas
"""

from par2vel.field import Field3D
from par2vel.camera import Scheimpflug
import numpy
import unittest

class testField3D(unittest.TestCase):
    def testgrids(self):
        cam1 = Scheimpflug((512,512))
        cam2 = Scheimpflug((1024,1024))
        cam1.set_calibration(numpy.pi/3,1/1000)
        cam2.set_calibration(-numpy.pi/4,1/500)
        field = Field3D([cam1,cam2])
        field.grid([25,20])
        X_1 = field.field2d[0].camera.x2X(field.field2d[0].xflat())
        X_2 = field.field2d[1].camera.x2X(field.field2d[1].xflat())
        for i in range(500):
            for k in range(3):
                self.assertAlmostEqual(X_1[k,i],X_2[k,i])
                
    def testpartial(self):
        cam1 = Scheimpflug((32,32))
        cam2 = Scheimpflug((32,32))
        cam1.set_calibration(numpy.pi/3,1/100)
        cam2.set_calibration(-numpy.pi/3,1/100)
        field = Field3D([cam1,cam2])
        field.grid([2,2])
        field.dxdX()
        

if __name__ == '__main__':
    unittest.main()