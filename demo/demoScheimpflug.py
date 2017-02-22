# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:25:59 2017

@author: Lukas Høghøj

Demonstration on how to use the Scheimpflug with the par2vel toolbox
"""
import os
os.system('cd ..')
import numpy as np
import matplotlib.pyplot as plt
from par2vel.camera import Scheimpflug
from par2vel.field import Field2D
from par2vel.artimage import ArtImage, constUfunc, OseenUfunc
from par2vel.piv2d import fftdx

cam = Scheimpflug((512,512))
cam.set_calibration(np.pi/3,1/1000) #M between 0 & 1
cam2 = Scheimpflug((512,512))
cam2.set_calibration(-np.pi/3,1/1000)
ai = ArtImage(cam)
ai2 = ArtImage(cam2)
ai.random_particles(0.02)
ai.displace_particles(constUfunc(np.array([0.1,0.05,0.001])), 1)
#ai.displace_particles(OseenUfunc(1,18,[0,0,0.0]), 0.0001)
ai2.X = ai.X
ai.generate_images()
ai2.generate_images()
iagrid = Field2D(cam)
iagrid2 = Field2D(cam2)
iagrid.squarewindows(32, 0.5)
iagrid2.squarewindows(32, 0.5)
fftdx(ai.Im[0], ai.Im[1], iagrid)
fftdx(ai2.Im[0], ai2.Im[1], iagrid2)
"""
plt.figure(1)
plt.imshow(ai.Im[0], cmap='gray')
plt.figure(2)
plt.imshow(ai.Im[1], cmap='gray')
plt.figure(3)
plt.imshow(ai2.Im[0], cmap='gray')
plt.figure(4)
plt.imshow(ai2.Im[1], cmap='gray')
"""
plt.figure(1)
plt.quiver(iagrid.x[0,:,:],iagrid.x[1,:,:],iagrid.dx[0,:,:],iagrid.dx[1,:,:])
plt.axis('image')
# plt.show()
plt.figure(2)
plt.quiver(iagrid2.x[0,:,:],iagrid2.x[1,:,:],iagrid2.dx[0,:,:],iagrid2.dx[1,:,:])
plt.axis('image')
plt.show()