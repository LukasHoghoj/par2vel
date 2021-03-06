#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:52:31 2017

@author: Lukas Hoghoj

Demonstration on how to perform stereo PIV
"""
# Get time to run program (tool to optimize)
import time
t_start = time.time()
#### start ####
import numpy as np
import matplotlib.pyplot as plt
from par2vel.camera import Scheimpflug, Pinhole, Third_order
from par2vel.field import Field3D
from par2vel.artimage import ArtImage, constUfunc, OseenUfunc, X_square, X_U
from par2vel.piv2d import fftdx
#from par2vel.piv3d import piv_camplane, stereo

# Creating Scheimpflug cameras as reference for third order or pinhole cameras
cam_scheim = Scheimpflug((512,512))
cam_scheim.set_calibration(np.pi/3,1/1000) #M between 0 & 1
cam2_scheim = Scheimpflug((512,512))
cam2_scheim.set_calibration(-np.pi/3,1/1000)

X = np.mgrid[-2:2:0.1,-2:2:0.1,-0.001:0.001:0.001]
X = X.reshape((3,-1))
x_1 = cam_scheim.X2x(X)
x_2 = cam2_scheim.X2x(X)
"""
cam = Pinhole((512, 512))
cam.calibration(x_1, X)
cam2 = Pinhole((512, 512))
cam2.calibration(x_2, X)
"""
cam = Third_order((512, 512))
cam.calibration(x_1, X)
cam2 = Third_order((512, 512))
cam2.calibration(x_2, X)


ai = ArtImage(cam)
ai2 = ArtImage(cam2)
ai.random_particles(0.02)
# Make new 3D displacement functions!!!
ai.displace_particles(X_U([0.005,0.002,0.001],[0,0,0]),1)
# ai.displace_particles(X_square([0.05,0.05,0.05],[0,0,0]),1)
# ai.displace_particles(constUfunc(np.array([0.1,0.05,0.05])), 1)
# ai.displace_particles(OseenUfunc(0.1,0.1,[0,0,0.0]), 1)
ai2.X = ai.X
ai.generate_images()
ai2.generate_images()
iagrid = Field3D([cam,cam2])
iagrid.grid([20,25])
fftdx(ai.Im[0],ai.Im[1],iagrid.field2d[0])
fftdx(ai2.Im[0],ai2.Im[1],iagrid.field2d[1])
t_stereo = time.time()
iagrid.dxdX()
iagrid.stereo()
print("Execution time for stereo part was %s seconds" % (time.time()-t_stereo))
imshowrange = iagrid.X_corners_rectangle.reshape(4)

plt.figure(1)
plt.imshow(ai.Im[0], cmap='gray')
plt.figure(2)
plt.imshow(ai.Im[1], cmap='gray')
plt.figure(3)
plt.imshow(ai2.Im[0], cmap='gray')
plt.figure(4)
plt.imshow(ai2.Im[1], cmap='gray')
plt.figure(1)
plt.quiver(iagrid.field2d[0].x[0,:,:],iagrid.field2d[0].x[1,:,:],\
           iagrid.field2d[0].dx[0,:,:],iagrid.field2d[0].dx[1,:,:])
plt.axis('image')
plt.figure(2)
plt.quiver(iagrid.field2d[1].x[0,:,:],iagrid.field2d[1].x[1,:,:],\
           iagrid.field2d[1].dx[0,:,:],iagrid.field2d[1].dx[1,:,:])
plt.axis('image')
plt.figure(3)
plt.quiver(iagrid.X[0,:,:],iagrid.X[1,:,:],iagrid.dX[0,:,:],iagrid.dX[1,:,:])
plt.imshow(iagrid.dX[2,:,:],cmap = 'cool',interpolation = 'bilinear' ,\
           extent=imshowrange,origin='lower')
plt.colorbar(orientation = 'horizontal')
plt.show()
print("Execution time was %s seconds" % (time.time()-t_start))