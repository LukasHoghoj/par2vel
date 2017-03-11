import unittest
import numpy
from par2vel.camera import Camera , Scheimpflug
from par2vel.field import Field3D
from par2vel.piv3d import stereo
from par2vel.artimage import X_U




class StereoTest(unittest.TestCase):
	def testStereo(self):
		"""This unittest makes a displacement in the obect plane,
		transforms it to both cameras and then uses the stereo 
		function to find the displacement in object plane, both
		vectors are then compared"""
		cam1 = Scheimpflug((512,512))
		cam2 = Scheimpflug((512,512))
		cam1.set_calibration(numpy.pi/3,1/200)
		cam2.set_calibration(-numpy.pi/3.1,1/250)
		field = Field3D([cam1,cam2])
		field.grid([3,4])
		dX_flat = field.getX_flat()/100
		dX = dX_flat.reshape(field.X.shape)
		X_flat = field.X.reshape((3,-1))
		field.field2d[0].dx = field.field2d[0].camera.dX2dx(X_flat,dX_flat).reshape((2,4,3))
		field.field2d[1].dx = field.field2d[1].camera.dX2dx(X_flat,dX_flat).reshape((2,4,3))
		stereo(field)
		numpy.testing.assert_array_almost_equal(dX,field.dX)

if __name__ == '__main__':
    unittest.main(exit = False)