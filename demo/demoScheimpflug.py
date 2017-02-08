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
cam.theta = 0
ai = ArtImage(cam)
#ai.random_particles(0.02)
dx =np.array([5.1,2.2])
#ai.displace_particles(constUfunc(dx), 1)
#ai.displace_particles(OseenUfunc(200,8,[256,256,0.0]), 1)
ai.generate_images()
iagrid = Field2D(cam)
iagrid.squarewindows(32, 0.5)
#fftdx(ai.Im[0], ai.Im[1], iagrid)
"""plt.figure(1)
plt.imshow(ai.Im[0], cmap='gray')
plt.figure(2)
plt.imshow(ai.Im[1], cmap='gray')
plt.figure(3)
plt.quiver(iagrid.x[0,:,:],iagrid.x[1,:,:],iagrid.dx[0,:,:],iagrid.dx[1,:,:])
plt.axis('image')
plt.show()"""