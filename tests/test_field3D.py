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
        

if __name__ == '__main__':
    unittest.main()