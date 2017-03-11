import unittest
import numpy
from par2vel.camera import Camera , Scheimpflug
from par2vel.field import Field3D
from par2vel.piv3d import stereo
from par2vel.artimage import X_U

def dis_func(X)
	s = X.shape
	X.reshape((3,-1))
	dX = zeros(s)
	dX[0] = X[0]/100
	dX[1] = X[1]/100
	dX[2] = X[0]/100
	dX.reshape((3,s[2],s[2]))
	return dX


class StereoTest(unittest.TestCase):
	
	def TestStereo(self):
		cam1 = Scheimpflug((512,512))
		cam2 = Scheimpflug((512,512))
		cam1.set_calibration(numpy.pi/3,1/200)
		cam2.set_calibration(-numpy.pi/3.1,1/250)
		field = Field3D([cam1,cam2])
		field.grid([3,4])
		dX = dis_func(field.X)
		dXflat = dX.reshape((3,-1))
		X_flat = field.X.reshape((3,-1))
		field.field2d[0].dx = field.field2d[0].camera.dX2dx(X_flat,dX_flat).reshape((2,4,3))
		field.field2d[1].dx = field.field2d[1].camera.dX2dx(X_flat,dX_flat).reshape((2,4,3))
		stereo(field)
		self.AssertAlmostEqual(dX,field.dX)