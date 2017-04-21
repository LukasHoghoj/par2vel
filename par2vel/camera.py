""" Camera setup and camera models used in par2vel """
# Copyright Knud Erik Meyer 2017
# Open source software under the terms of the GNU General Public License ver 3

import numpy
import re
from PIL import Image

class Camera(object):
    """Base class for camera models"""

    # valid keywords in camera file - used as variables in the class
    keywords = {'pixels': 'Image size in pixel (two integers: nrow,ncolumn)',
                'pixel_pitch': 'Pixel pitch in meters (two floats, follow x)',
                'fill_ratio': 'Fraction of active pixel area in x ' + \
                              'and y (two floats)',
                'noise_mean': 'Mean noiselevel - fraction (float)',
                'noise_rms':  'RMS noiselevel - fraction (float)',
                'focal_length': 'Focal length in meters (float)',
                'f_number':   'F-number on lens (float)'}

    def __init__(self, pixels=None):
        """Define a camera, possibly by giving size in pixels"""
        # define camera calibration model (= class)
        self.model = 'base'
        # set some default values (may be overwritten by camera file)
        self.pixel_pitch = (1e-5, 1e-5)  # 10 micron
        self.fill_ratio = (1.0, 1.0)
        self.noise_mean = 0.0
        self.noise_rms = 0.0
        self.focal_length = 0.06
        # allow pixels to be changed at creation (mostly for tests)
        if pixels:
            self.pixels = pixels
        else:
            self.pixels = (512, 512) # default value
        assert len(self.pixels) == 2
        # set shape to pixels
        self.shape = self.pixels
        
        

    def set_physical_size(self):
        """Set a guess on dimensions in physical space"""
        from numpy import array
        # For this base class we just use pixel coordinates directly
        # size of a 1 pixel displacement in physical space
        self.dXpixel = 1.0   
        # width and height of physical region                                       
        # roughly correponding to image
        self.Xsize = array([self.pixels[1], self.pixels[0]])
        # intersection (roughtly) optical axis and physical plane
        self.Xopticalaxis = (self.Xsize - 1) * 0.5 

    def set_keyword(self, line):
        """Set a keyword using a line from the camera file"""
        # comment or blank line - do nothing
        if re.search('^\\s*#',line) or re.search('^\\s*$',line): 
            return
        # remove any trailing \r (files from windows used in linux)
        if line[-1]=='\r':
            line = line[0:-1]
        # find first word, allow leading whitespace
        firstword = re.findall('^\\s*\\w+',line) 
        if firstword:
            if firstword[0] in self.keywords:
                try:
                    exec('self.'+line) # assign as variable to camera
                except:
                    print('Error at the line:')
                    print(line)
                    print('The keyword',firstword[0],'should be:', end=' ')
                    print(self.keywords[firstword[0]])
                    raise Exception('Camera file error')
            elif firstword[0] == 'model':
                pass
            else:
                print('Error at the line:')
                print(line)
                print('The keyword:',firstword[0],'is not known')
                raise Exception('Camera file error')
        else: 
            print('Error at the line:')
            print(line)
            print('Line does not start with keyword or camera model')
            raise Exception('Camera file error')

    def read_camera(self, filename):
        """Read camera definition and/or calibration data"""
        # basic camera only reads camera keywords
        lines = open(filename).readlines()
        for line in lines:
            self.set_keyword(line)
        self.shape = self.pixels
        

    def save_keywords(self, f):
        """Saves defined keywords to an already open camera file f"""
        names = dir(self)
        for keyword in self.keywords:
            if keyword in names:
                #exec('value = self.' + keyword) # get the value
                value = eval('self.' + keyword)
                print(keyword, '=', value, file=f)

    def save_camera(self, filename):
        """Save camera definition and/or calibration data"""
        # basic camera only saves defined camera keywords
        f = open(filename,'w')
        f.write('# par2vel camera file\n')
        f.write("model = '{:}'\n".format(self.model))
        self.save_keywords(f)
      
    def set_calibration(self, calib):
        """Set calibration parameters"""
        # actually not needed in base class
        self.calib = calib

    def X2x(self, X):
        """Use camera model to get camera coordinates x
           from physical cooardinates X.
        """
        # this is simple model that simply assume samme coordinate system
        return X[0:2,:]

    def record_image(self, image, ijcenter, pitch):
        """Record an image given in physical space"""
        # ijcenter is image index where optical axis intersects image
        # pitch is pixel pitch of image in meters
        from scipy.interpolate import interp2d
        from numpy import arange, array, indices
        # make x for image sensor
        xtmp = indices(self.shape)
        x = array([xtmp[1,:],xtmp[0,:]], dtype=float)
        # get corresponding coordinates in object space
        X = self.x2X(x.reshape(2,-1))
        # find coordinates corresponding to supplied image
        xim = X[0:2,:] / pitch + [ [ijcenter[1]], [ijcenter[0]] ]
        # find nearest center
        i = xim[1,:].round().astype(int)
        j = xim[0,:].round().astype(int)
        newimage = image[i,j]
        newimage = newimage.reshape(self.shape)
        return newimage  
        
    def dX2dx(self, X, dX):
        """Transform displacements in obejct plane to image plane
        input as 2 3D vectors"""
        dx = self.X2x(X + 0.5 * dX) - self.X2x(X - 0.5 * dX)
        return dx

    def dx2dX(self, x, dx):
        """Transform displacements in camera plane to image plane,
        assuming that dX[2] = 0 i.e. there is no displacement in the
        Z direction"""
        dX = self.x2X(x + 0.5 * dx) - self.x2X(x - 0.5 * dx)
        return dX

    def part(self, X, n = 8):
        """Compute the partial derivatives at a point"""
        from numpy import eye, repeat, tile
        len = X.shape[1]
        disp =  tile(eye(3) * 10**(-n), len)
        dxdX = (self.X2x(repeat(X, 3).reshape(3, 3 * len) + disp) -\
                self.X2x(repeat(X, 3).reshape(3, 3 * len))) *  10 ** n
        return dxdX

    def x2X(self, x):
        """Transform image plane coordinates to object plane, assuming Z = 0"""
        import numpy as np
        import scipy.optimize as opt
        # Create empty solution vector (3rd dimension will always stay 0 as assumed)
        X = np.zeros((3,x.shape[1]))
        XY_guess = np.zeros(2)
        for i in range(x.shape[1]):
            X[0 : 2, i] = opt.fsolve(self.X2x,XY_guess,x[:,i])
        return X

class One2One(Camera):
    """Camera model that assumes object coordinates = image coordinates"""
    # same functions as Camera, but adds inverse functions x2X and dx2dX
    def __init__(self, newshape=None):
        Camera.__init__(self,newshape)
        # define camera calibration model (= class)
        self.model = 'One2One'

    def x2X(self, x, z=0):
        """Find physical coordinates from image coordinates
           We assume that third physical coordinate z=0, but a different
           value can be given as a float.
        """
        # this is simple model that simply assume same coordinate system
        from numpy import ones, vstack
        dummy, n = x.shape
        X = vstack((x, z*ones((1,n))))
        return X

    def dx2dX(self, x, dx, z=0):
        """Transform displacement in pixel to physical displacement.
           The displacement is assumed to be at the z=0 plane in
           physical space.
        """
        # very simple model setting physical coordinates to image coord.
        from numpy import zeros, vstack
        dummy, n = dx.shape
        dX = vstack((dx, zeros((1,n))))
        return dX    

class Linear2d(Camera):
    """Camera model for simple 2D PIV using Z=0 always"""
    def __init__(self, newshape=None):
        Camera.__init__(self,newshape)
        # define camera calibration model (= class)
        self.model = 'Linear2d'

    def set_physical_size(self):
        """Set a guess on dimensions in physical space"""
        from numpy import array, sqrt 
        # intersection (roughtly) optical axis and physical plane
        x_center = (array([self.shape[1], self.shape[0]]) - 1) * 0.5
        x_center.shape = (2, 1)
        self.Xopticalaxis = self.x2X(x_center)
        # width and height of physical region roughly correponding to image
        x0 = array([-0.5, -0.5]).reshape((2,1))
        xmax = array([self.shape[1], self.shape[0]]) + x0
        self.Xsize = abs(self.x2X(xmax) - self.x2X(x0))  #FIX: not right shape
        # size of a 1 pixel displacement in physical space
        self.dXpixel = sqrt(((self.x2X(x_center) -
                              self.x2X(x_center + [[1],[0]]))**2).sum())

    def set_calibration(self, calib):
        """Set calibration parameters"""
        # Calibration is a 2x3 matrix
        assert calib.shape == (2,3)
        self.calib = calib

    def read_camera(self, filename):
        """Read camera definition and/or calibration data"""
        lines = open(filename).readlines()
        nlines = len(lines)
        n = 0
        while n < nlines:
            line = lines[n]
            # check for calibration data
            if line.lower().find('calibration') == 0:
                if line.lower().find('linear2d') > 0:
                    calib = numpy.array([
                        [float(x) for x in lines[n+1].split()],
                        [float(x) for x in lines[n+2].split()] ])
                    self.set_calibration(calib)
                    n += 2                
            else:
                self.set_keyword(line)
            n += 1
        self.shape = self.pixels

    def save_camera(self, filename):
        """Save camera definition and/or calibration data"""
        f = open(filename,'w')
        f.write('# par2vel camera file\n')
        f.write("model = '{:}'\n".format(self.model))
        # first save defined keywords
        self.save_keywords(f)
        # save calibration
        print('Calibration Linear2d', file=f)
        for row in self.calib:
            for number in row:
                print(number, end=' ', file=f)
            print(file=f)
        f.close()

    def X2x(self, X):
        """Use camera model to get camera coordinates x
           from physical cooardinates X.
        """
        from numpy import vstack, dot, ones
        # append row of ones below first two rows of X
        ni, nj = X.shape
        Xone = vstack((X[0:2,:],ones(nj)))
        # transformation is found by multiplying with calib matrix
        x = dot(self.calib,Xone)
        return x

    def x2X(self, x, z=0):
        """Find physical coordinates from image coordinates
           We assume that third physical coordinate z=0, providing another
           value will have not effect in Linear2D.
        """
        from numpy import vstack, dot, zeros
        from numpy.linalg import inv
        ni, nj = x.shape
        calibinv = inv(self.calib[:,0:2])
        X = dot(calibinv,(x - self.calib[:,2].reshape((-1,1))))
        X = vstack((X, zeros(nj)))
        return X

    def dx2dX(self, x, dx, z=0):
        """Transform displacement in pixel to physical displacement.
           The displacement is assumed to be at the z=0 plane in
           physical space.
        """
        # very simple model setting physical coordinates to image coord.
        from numpy import dot
        from numpy.linalg import inv
        calibinv = inv(self.calib[:,0:2])
        dX = dot(calibinv,dx)
        return dX

class Linear3d(Camera):
    """Camera model using Direct Linear Transform (DFT)"""
    def __init__(self, newshape=None):
        Camera.__init__(self,newshape)
        # define camera calibration model (= class)
        self.model = 'Linear3d'

##    def set_physical_size(self):
##        """Set a guess on dimensions in physical space"""
##        from numpy import array, sqrt 
##        # intersection (roughtly) optical axis and physical plane
##        x_center = (array([self.shape[1], self.shape[0]]) - 1) * 0.5
##        x_center.shape = (2, 1)
##        self.Xopticalaxis = self.x2X(x_center)
##        # width and height of physical region roughly correponding to image
##        x0 = array([-0.5, -0.5]).reshape((2,1))
##        xmax = array([self.shape[1], self.shape[0]]) + x0
##        self.Xsize = abs(self.x2X(xmax) - self.x2X(x0))  #FIX: not right shape
##        # size of a 1 pixel displacement in physical space
##        self.dXpixel = sqrt(((self.x2X(x_center) -
##                              self.x2X(x_center + [[1],[0]]))**2).sum())

    def set_calibration(self, calib):
        """Set calibration parameters"""
        # Calibration is a 3x4 matrix
        assert calib.shape == (3,4)
        self.calib = calib

    def read_camera(self, filename):
        """Read camera definition and/or calibration data"""
        lines = open(filename).readlines()
        nlines = len(lines)
        n = 0
        while n < nlines:
            line = lines[n]
            # check for calibration data
            if line.lower().find('calibration') == 0:
                if line.lower().find('linear3d') > 0:
                    calib = numpy.array([
                        [float(x) for x in lines[n+1].split()],
                        [float(x) for x in lines[n+2].split()],
                        [float(x) for x in lines[n+3].split()] ])
                    self.set_calibration(calib)
                    n += 3                
            else:
                self.set_keyword(line)
            n += 1
        self.shape = self.pixels

    def save_camera(self, filename):
        """Save camera definition and/or calibration data"""
        f = open(filename,'w')
        f.write('# par2vel camera file\n')
        f.write("model = '{:}'\n".format(self.model))
        # first save defined keywords
        self.save_keywords(f)
        # save calibration
        print('Calibration Linear3d', file=f)
        for row in self.calib:
            for number in row:
                print(repr(number), end=' ', file=f)
            print(file=f)
        f.close()

    def X2x(self, X):
        """Use camera model to get camera coordinates x
           from physical cooardinates X.
        """
        from numpy import vstack, dot, ones
        # append row of ones below first two rows of X
        ni, nj = X.shape
        Xone = vstack((X,ones(nj)))
        # transformation 
        k = dot(self.calib,Xone)
        # apply perspective correction
        x = k[0:2,:] / k[2,:]
        return x



class Scheimpflug(Camera):
    """Camera model for simple Scheimpflug camera"""
    # this camera assumes the coordinatesystem to have origin on optical axis
    def __init__(self, newshape=None):
        Camera.__init__(self,newshape)

    def set_physical_size(self):
        """Set a guess on dimensions in physical space"""
        from numpy import array 
        # intersection (roughtly) optical axis and physical plane
        # - in this case coordinate systems origin in on optical axis
        self.Xopticalaxis = array([[0.0], [0.0], [0.0]])
        # width and height of physical region roughly correponding to image
        size_camerachip = p * array([self.pixels[1], self.pixels[0]])
        self.Xsize = size_camerachip / self.M
        # size of a 1 pixel displacement in physical space
        self.dXpixel = p / M

    def set_calibration(self, theta, M):
        """Set calibration parameters (theta in radians)"""
        # Calibration uses the following parameters
        self.M = M           # magnification
        self.theta = theta   # angle between optical axis and object plane
        # note that focal_length and pixel_pitch must be defined in camera
                             
    def read_camera(self, filename):
        """Read camera definition and/or calibration data"""
        lines = open(filename).readlines()
        nlines = len(lines)
        n = 0
        while n < nlines:
            line = lines[n]
            # check for calibration data
            if line.lower().find('calibration') == 0:
                if line.lower().find('scheimpflug') > 0:
                    fields = lines[n+1].split()
                    self.set_calibration(float(fields[0]), float(fields[1]))
                    n += 1                
            else:
                self.set_keyword(line)
            n += 1
        self.shape = self.pixels

    def save_camera(self, filename):
        """Save camera definition and/or calibration data"""
        f = open(filename,'w')
        f.write('# par2vel camera file\n')
        f.write("model = '{:}'\n".format(self.model))
        # first save defined keywords
        self.save_keywords(f)
        # save calibration
        f.write('Calibration Scheimpflug\n')
        f.write('{:} {:}\n'.format(self.theta, self.M))
        f.close()

    def X2x(self, X):
        """Use camera model to get camera coordinates x
           from physical cooardinates X.
        """
        from numpy import cos, sin, tan, arctan, sqrt, vstack
        # find angles and distances
        theta = self.theta
        alpha = arctan(self.M * tan(theta))
        p = self.pixel_pitch[0] # use pitch in x0 direction
        a = self.focal_length * (self.M + 1) / self.M
        b = self.M * a
        xcenter0 = 0.5 * self.pixels[1] - 0.5   # note model for camera coord. 
        xcenter1 = 0.5 * self.pixels[0] - 0.5
        # find unitvector elements
        uox = sin(theta)
        uoy = cos(theta)
        uix = p * sin(alpha)
        uiy = -p * cos(alpha)
        # arrays with coordinates in physical space
        r = X[0,:]
        s = X[2,:]
        # find corresponding image coordinate
        t = - b * (r * uoy + s * uox) / ( uiy * (a + r * uox - s * uoy)
                                         - uix * (r * uoy + s * uox) )
        # local magnification
        m = sqrt( ((-b + t * uix)**2 + (t * uiy)**2)   /
                  ((a + r * uox - s * uoy)**2 + (r * uoy + s * uox)**2) )
        # make x matrix
        x0 = t + xcenter0
        x1 = -(m * X[1,:] / self.pixel_pitch[1]) + xcenter1
        x = vstack((x0, x1))
        return x

    def x2X(self, x):
        """Use camera model to get physcial coordinates X from c
           camera coordinates x (assuming X[2]=0)
        """
        # using solution from PIV book by Raffel et al (2007), page 215
        from numpy import cos, sin, tan, arctan, vstack, zeros, sqrt
        ni, nj = x.shape
        # find angles and distances
        theta = self.theta
        alpha = arctan(self.M * tan(theta))
        p = self.pixel_pitch[0] # use pitch in x0 direction
        a = self.focal_length * (self.M + 1) / self.M
        b = self.M * a
        xcenter0 = 0.5 * self.pixels[1] - 0.5   # note model for camera coord. 
        xcenter1 = 0.5 * self.pixels[0] - 0.5
        # find unitvector elements
        uox = sin(theta)
        uoy = cos(theta)
        uix = p * sin(alpha)
        uiy = -p * cos(alpha)
        # arrays with image coordinates (relative to center)
        t = x[0,:] - xcenter0
        # find corresponding object coordinate
        r = t * uiy * a / (-uoy * b + t * uoy * uix - t * uox * uiy)
        # local magnification
        m = sqrt( ((-b + t * uix)**2 + (t * uiy)**2)   /
                  ((a + r * uox)**2 + (r * uoy)**2) )
        # make physical coordinates
        X0 = r
        X1 = -self.pixel_pitch[1] * (x[1,:] - xcenter1) / m
        X2 = zeros((1,nj))
        X = vstack((X0, X1, X2))
        return X

    def dx2dX(self, x, dx, z=0):
        """Transform displacement in pixel to physical displacement.
           The displacement is assumed to be at the z=0 plane in
           physical space (other values cannot be used).
        """
        # very simple model setting physical coordinates to image coord.
        from numpy import zeros, vstack
        assert z == 0
        dX = self.x2X(x + 0.5 * dx) - self.x2X(x - 0.5 * dx)
        return dX      


    def dX2dx(self, X, dX):
        """Use camera models to transform displacement in object coordinates
        to displacement in image coordinates"""
        dx = self.X2x(X + 0.5 * dX) - self.X2x(X - 0.5 * dX)
        return dx

class Pinhole(Camera):
    """ Camera model for pinhole model"""
    def __init__(self, newshape = None):
        Camera.__init__(self,newshape)
        # Define camera model
        self.model = 'Pinhole'
    
    def Calibrate_Pinhole(self,X, x, C):
        """Function to calibrate camera with respect to the pinhole model. The
        function takes the input Calibrate_Pinhole([X,Y,Z],[x,y]); where X,Y and Z
        are the coordinates in object space and x and y are their respectevely
        corresponding coordinates in the image plane. The argument C is a guess
        for the camera matrix"""
        import numpy as np
        from numpy.linalg import inv
        from scipy.linalg import lstsq
        import scipy.optimize as opt
        import scipy as sp
        assert X.shape[1] == x.shape[1]
        # Maximum number of iterations:
        ite_max = 50
        # Maximum error:
        err_max = 0.5
        x_1 = np.vstack((x,np.ones((1,x.shape[1]))))
        # Length of given data
        len = X.shape[1]
        # Add ones as 4th dimension to physical coordinates
        X_p = np.vstack((X, np.ones(len)))
        x_d = np.zeros(x.shape)
        # Compute normalized coordinates corrected for distorted with camera
        # matrix guess
        x_d[0, :] = (x[0, :] - C[0, 2]) / C[0, 0]
        x_d[1, :] = (x[1, :] - C[1, 2]) / C[1, 1]
        # Find roation and translation matrix, that fit's the best for transforming
        # physical coordinates to nondistorted camera plane
        R = self.Rotation_T(x_d, X)
        # Find resulting distorted normalized coordinates
        X_C = np.dot(R, X_p)
        x_n = np.zeros((2, len))
        x_n[0] = X_C[0] / X_C[2]
        x_n[1] = X_C[1] / X_C[2]
        # Find distortion constants that fit the best
        #k1, k2, k3, p1, p2 = self.Distortion(x_d, x_n)
        dis = self.Distortion(x_d, x_n)
        # Correct for distortion with the new constants
        x_d = self.dis_method(x_n, dis)
        # Find new camera matrix, by using the real camera coordinates and the normalized
        # coordinates, that are corrected for distortion
        C = self.Cam_Matrix(x, x_d)
        x_p = np.dot(C, np.vstack((x_d, np.ones(len))))
        # Compute average error:
        err = np.mean(np.sqrt((x_p[0] - x[0]) ** 2 + (x_p[1] - x[1]) ** 2))
        ite = 0
        while err >= err_max and ite < ite_max:
            # Recompute through the system to find the error:
            x_d = (inv(np.vstack((C,np.array([0,0,1])))).dot(x_1))[0 : 2, :]
            # Compute distortion backwards
            for i in range(len):
                x_n[:,i] = opt.fsolve(self.dis_method, x_d[:,i],args = ( dis, x_d[:,i]))
            # New matrix for rotation/transtlation
            R = self.Rotation_T(x_d, X)
            # Compute new distorted coordinates
            X_C = np.dot(R, X_p)
            x_n[0] = X_C[0] / X_C[2]
            x_n[1] = X_C[1] / X_C[2]
            # New distortion coefficients:
            dis = self.Distortion(x_d, x_n)
            x_d = self.dis_method(x_n, dis)
            C = self.Cam_Matrix(x, x_d)
            x_p = np.dot(C, np.vstack((x_d, np.ones(len))))
            err = np.mean(np.sqrt((x_p[0] - x[0]) ** 2 + (x_p[1] - x[1]) ** 2))
            ite += 1
            print(ite)
        print(err)
        k1, k2, k3, p1, p2 = dis
        self.R = R.astype(numpy.float64)
        self.C = C.astype(numpy.float64)
        self.k1 = k1.astype(numpy.float64)
        self.k2 = k2.astype(numpy.float64)
        self.k3 = k3.astype(numpy.float64)
        self.p1 = p1.astype(numpy.float64)
        self.p2 = p2.astype(numpy.float64)

    def dis_method(self, x_n, dis, dif = 0):
        """Help function for the pinhole calibration"""
        assert x_n.shape[0] == 2
        from numpy import vstack, zeros, ndim
        a = 0
        if x_n.shape[0] == 2 and ndim(x_n) == 1:
            x_n = x_n.reshape(2,1)
            a = 1
        len = x_n.shape[1]
        k1, k2, k3, p1, p2 = dis
        x_d = zeros((2, len))
        r = x_n[0 , :] ** 2 + x_n[1 , :] ** 2
        x_d[0 , :] = x_n[0 , :] * (1 + k1 * r + k2 * r ** 2 + k3 * r ** 3)\
                     + 2 * p1 * x_n[0 , :] * x_n[1 , :] + p2 * (r + 2 * x_n[0 , :] ** 2)
        x_d[1 , :] = x_n[1 , :] * (1 + k1 * r + k2 * r ** 2 + k3 * r ** 3)\
                     + p1 * (r + 2 * x_n[1 , :] ** 2) + 2 * p2 * x_n[0 , :] * x_n[1 , :]
        if a == 1:
            x_d = x_d.reshape(2)
        return x_d - dif
        
    def Distortion(self,x_d, x_n):
        """Function returns the distortion cooeficients for given distorted and 
        normalized coordinates"""
        import numpy as np
        from scipy.linalg import lstsq
        import scipy as sp
        assert x_d.shape == x_n.shape
        len = x_d.shape[1]
        lhs = np.zeros(len * 2)
        i = np.arange(len * 2)
        lhs[i % 2 == 0] = x_d[0, :] - x_n[0, :]
        lhs[i % 2 == 1] = x_d[1, :] - x_n[1, :]
        rhs = np.zeros((len * 2, 5))
        r = (x_n[0, :] ** 2 + x_n[1, :] ** 2)
        rhs[:, 0][i % 2 == 0] = x_n[0, :] * r
        rhs[:, 0][i % 2 == 1] = x_n[1, :] * r
        rhs[:, 1][i % 2 == 0] = x_n[0, :] * r ** 2
        rhs[:, 1][i % 2 == 1] = x_n[1, :] * r ** 2
        rhs[:, 2][i % 2 == 0] = x_n[0, :] * r ** 3
        rhs[:, 2][i % 2 == 1] = x_n[1, :] * r ** 3
        rhs[:, 3][i % 2 == 0] = 2 * x_n[0, :] * x_n[1, :]
        rhs[:, 3][i % 2 == 1] = r + 2 * x_n[1, :] ** 2 
        rhs[:, 4][i % 2 == 0] = r + 2 * x_n[0, :] ** 2
        rhs[:, 4][i % 2 == 1] = 2 * x_n[0, :] * x_n[1, :]
        
        dis = lstsq(rhs,lhs)[0]
        return dis

    def Rotation_T(self,x_n, X_p):
        """Function that finds the rotation and translation parameters """
        import numpy as np
        from scipy.linalg import lstsq
        import scipy as sp
        assert x_n.shape[1] == X_p.shape[1]
        len = x_n.shape[1]
        lhs = np.zeros(len * 2)
        rhs = np.zeros((len * 2, 11))
        i = np.arange(len * 2)
        lhs[i % 2 == 0] = x_n[0, :]
        lhs[i % 2 == 1] = x_n[1, :]
        rhs[:, 0 : 3][i % 2 == 0] = X_p.T
        rhs[:, 3][i % 2 == 0] = 1
        rhs[:, 0 : 4][i % 2 == 1] = 0
        rhs[:, 4 : 8][i % 2 == 0] = 0
        rhs[:, 4 : 7][i % 2 == 1] = X_p.T
        rhs[:, 7][i % 2 == 1] = 1
        rhs[:, 8][i % 2 == 0] = - x_n[0, :] * X_p[0, :]
        rhs[:, 8][i % 2 == 1] = - x_n[1, :] * X_p[0, :]
        rhs[:, 9][i % 2 == 0] = - x_n[0, :] * X_p[1, :]
        rhs[:, 9][i % 2 == 1] = - x_n[1, :] * X_p[1, :]
        rhs[:, 10][i % 2 == 0] = - x_n[0, :] * X_p[2, :]
        rhs[:, 10][i % 2 == 1] = - x_n[1, :] * X_p[2, :]
        #rhs[:, 11][i % 2 == 0] = - x_n[0, :]
        #rhs[:, 11][i % 2 == 1] = - x_n[1, :]
        RT = lstsq(rhs,lhs)[0]
        R = np.zeros((3,4))
        R[0] = RT[0 : 4]
        R[1] = RT[4 : 8]
        R[2, 0 : 3] = RT[8 : 12]
        R[2, 3] = 1
        return R

    def Cam_Matrix(self,x_p, x_d):
        """Function that optimizes the camera matrix, such that it fits the best
        """
        import numpy as np
        from scipy.linalg import lstsq
        import scipy as sp
        assert x_p.shape == x_d.shape
        len = x_p.shape[1]
        lhs = np.zeros(len * 2)
        i = np.arange(len * 2)
        rhs = np.zeros((len * 2, 4))
        lhs[i % 2 == 0] = x_p[0, :]
        lhs[i % 2 == 1] = x_p[1, :]
        rhs[:, 0][i % 2 == 0] = x_d[0, :]
        rhs[:, 0][i % 2 == 1] = 0
        rhs[:, 1][i % 2 == 0] = 0
        rhs[:, 1][i % 2 == 1] = x_d[1, :]
        rhs[:, 2][i % 2 == 0] = 1
        rhs[:, 2][i % 2 == 1] = 0
        rhs[:, 3][i % 2 == 0] = 0
        rhs[:, 3][i % 2 == 1] = 1

        f = lstsq(rhs, lhs)[0]
        Cam_M = np.zeros((2,3))
        Cam_M[0, 0] = f[0]
        Cam_M[1, 1] = f[1]
        Cam_M[0, 2] = f[2]
        Cam_M[1, 2] = f[3]

        return Cam_M

    def set_calibration(self, R, dis, C):
        """Set calibration for Pinhole camera model, to be caled as
        camera.set_calibration(R, dis, C), where R is the rotation and
        translation matrix, dis the 5D array with the lens distortion
        coefficients and C the camera matrix"""
        self.R = R
        self.k1 = dis[0]
        self.k2 = dis[1]
        self.k3 = dis[2]
        self.p1 = dis[3]
        self.p2 = dis[4]
        self.C = C

    def calibration(self, x, X, filename = False):
        """Calibrate camera and save calibration if a filename is in the input
        """
        from numpy import array
        C_guess  = array([[self.focal_length, 0, self.shape[0] / 2],[0, self.focal_length, self.shape[1] / 2]])
        self.Calibrate_Pinhole(X, x, C_guess)
        if filename != False:
            self.save_camera(filename)
    
    def save_camera(self, filename):
        """Save camera definition and/or calibration data"""
        from numpy import array
        f = open(filename,'w')
        f.write('# par2vel camera file\n')
        f.write("model = '{:}'\n".format(self.model))
        # first save defined keywords
        self.save_keywords(f)
        # save calibration
        print('Calibration Pinhole (Rotation matrix, distortion coefficients & Camera matrix)', file = f)
        for row in self.R:
            for number in row:
                print(repr(number), end=' ', file = f)
            print(file=f)
        dis = array([self.k1, self.k2, self.k3, self.p1, self.p2])
        for number in dis:
            print(repr(number), end = ' ', file = f)
        print(file = f)
        for row in self.C:
            for number in row:
                print(repr(number), end = ' ', file = f)
            print(file = f)
        f.close()
        
    def read_camera(self, filename):
        """This function reads the calibration file, that can is created if a filename
        input exists in the set_calibration function
        """
        lines = open(filename).readlines()
        nlines = len(lines)
        n = 0
        while n < nlines:
            line = lines[n]
            # check for calibration data
            if line.lower().find('calibration') == 0:
                if line.lower().find('pinhole') > 0:
                    R = numpy.array([
                        [float(x) for x in lines[n+1].split()],
                        [float(x) for x in lines[n+2].split()],
                        [float(x) for x in lines[n+3].split()] ])
                    dis = numpy.array([float(x) for x in lines[n+4].split()])
                    C = numpy.array([[float(x) for x in lines[n+5].split()],
                                     [float(x) for x in lines[n+6].split()]])
                    self.set_calibration(R,dis,C)
                    n += 6                
            else:
                self.set_keyword(line)
            n += 1
        self.shape = self.pixels

    def X2x(self,X, x_solve = 0):
        """Transformation from object to camera plane, input is a 3D vector (X,Y,Z),
        output a 2D vector (x,y)"""
        import numpy as np
        a = 0
        if X.shape[0] == 2 and np.ndim(X.shape) == 1:
            X = np.vstack((X.reshape(2,1),np.zeros(1)))
            a = 1
        len = X.shape[1]
        X_C = np.dot(self.R , np.vstack((X, np.ones(len))))
        x_n = np.zeros((2, len))
        if any(X_C[2] == 0):
            pos = np.argwhere(X_C[2] == 0)
            X_C[2, pos] = 0.01
            print('Correction reqired in X2x')
        x_n[0] = X_C[0] / X_C[2]
        x_n[1] = X_C[1] / X_C[2]
        x_d = np.zeros((2, len))
        r = x_n[0 , :] ** 2 + x_n[1 , :] ** 2
        x_d[0 , :] = x_n[0 , :] * (1 + self.k1 * r + self.k2 * r ** 2 + self.k3 * r ** 3)\
                     + 2 * self.p1 * x_n[0 , :] * x_n[1 , :] + self.p2 * (r + 2 * x_n[0 , :] ** 2)
        x_d[1 , :] = x_n[1 , :] * (1 + self.k1 * r + self.k2 * r ** 2 + self.k3 * r ** 3)\
                     + self.p1 * (r + 2 * x_n[1 , :] ** 2) + 2 * self.p2 * x_n[0 , :] * x_n[1 , :]
        x= np.dot(self.C , np.vstack((x_d, np.ones(len))))
        if a == 1:
            x = x.reshape(2)
        return x - x_solve

    def x2X(self, x):
        """Transformation from camera plane to object space. As the equation would be 
        underdefined, it is assumed that X[2] = 0"""
        import numpy as np
        from numpy.linalg import solve, inv
        import scipy.optimize as opt
        # Create empty solution vector (3rd dimension will always stay 0 as assumed)
        X = np.zeros((3,x.shape[1]))
        # First transform the image coordinates to physical plane (assuming no distortion)
        # in order to set up a guess
        x_p = np.vstack((x,np.ones((1,x.shape[1]))))
        x_d = (inv(np.vstack((self.C,np.array([0,0,1])))).dot(x_p))[0 : 2, :]
        base_rhs =  self.R[0 : 2, 0 : 2]
        for i in range(x.shape[1]):
            lhs = x_d[:, i] * self.R[2, 3] - self.R[0 : 2, 3]
            rhs = np.zeros((2,2))
            rhs[0, :] = base_rhs[0, :] - self.R[2, 0:2] * x_d[0, i]
            rhs[1, :] = base_rhs[1, :] - self.R[2, 0:2] * x_d[1, i]
            XY_guess = solve(rhs, lhs).reshape(2,1)
            X[0 : 2, i] = opt.fsolve(self.X2x,XY_guess[0:2],x[:,i])
        return X

class Third_order(Camera):
    """Third order camera model"""
    def __init__(self, newshape = None):
        Camera.__init__(self,newshape)
        # Define camera model
        self.model = 'Third_order'
    
    def set_calibration(self, calib):
        """Set calibration from given 2z20 array, that is given as input to
        this function"""
        assert calib.shape == (2,20)
        self.calib = calib

    def calibration(self, x, X, filename = False):
        """Calibrates the third order camera model, by using the input camera
        and object plane coordinates"""
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import lsqr
        from numpy import zeros, arange, array, tile, ones
        assert x.shape[1] == X.shape[1]
        assert X.shape[0] == 3
        assert x.shape[0] == 2
        len = x.shape[1]
        # Empty sparse matrix for the right hand side of the equation system,
        # which will give the calibration array
        rhs = zeros((2 * len, 40))
        # Empty array for left hand side
        lhs = zeros(2 * len)
        # Indices to assign values to the sparse matrix and the array
        index = arange(2 * len)
        # Assign image plane coordinates to left hand side of the equation system
        lhs[index % 2 == 0] = x[0]
        lhs[index % 2 == 1] = x[1]
        # Setting up the 3rd order values in a 20D array
        third = array([X[0], X[1], X[2], X[0] * X[1], X[0] * X[2], X[1] * X[2],
                       X[0] ** 2, X[1] ** 2, X[2] ** 2, X[0] ** 2  * X[1], X[0] ** 2
                       * X[2], X[1] ** 2 * X[0], X[1] ** 2 * X[2], X[2] ** 2 * X[0],
                       X[2] ** 2 * X[1], X[0] ** 3, X[1] ** 3, X[2] ** 3, X[0] * X[1]
                       * X[2], ones(len)]).T
        rhs[:, 0 : 20][index % 2 == 0] = third
        rhs[:, 20 : 40][index % 2 == 1] = third
        calib = lsqr(rhs, lhs)[0].reshape(2,20)
        self.set_calibration(calib)
        if filename != False:
            self.save_camera(filename)

    def save_camera(self, filename):
        """Save camera definition and/or calibration data"""
        from numpy import array
        f = open(filename,'w')
        f.write('# par2vel camera file\n')
        f.write("model = '{:}'\n".format(self.model))
        # first save defined keywords
        self.save_keywords(f)
        # save calibration
        print('Calibration third order', file = f)
        for row in self.calib:
            for number in row:
                print(repr(number), end=' ', file = f)
            print(file=f)

    def read_camera(self, filename):
        """This function reads the calibration file, that can is created if a filename
        input exists in the set_calibration function
        """
        lines = open(filename).readlines()
        nlines = len(lines)
        n = 0
        while n < nlines:
            line = lines[n]
            # check for calibration data
            if line.lower().find('calibration') == 0:
                if line.lower().find('third order') > 0:
                    calib = numpy.array([
                        [float(x) for x in lines[n+1].split()],
                        [float(x) for x in lines[n+2].split()] ])
                    self.set_calibration(calib)
                    n += 2
            else:
                self.set_keyword(line)
            n += 1
        self.shape = self.pixels

    def X2x(self, X, dif = 0):
        """Transform object plane coordinates to image plane, if a 2D array is
        given as input it is assumed that Z = 0 for all points in the array.
        The optional argument dif can be used to get the difference to a given 
        point in image plane"""
        from numpy import vstack, zeros, array, dot, ndim, ones
        assert (X.shape[0] == 2 and ndim(X.shape) == 1) or X.shape[0] == 3
        a = 0
        if X.shape[0] == 2 and ndim(X.shape) == 1:
            X = vstack((X.reshape(2,1),0))
            a = 1
        len = X.shape[1]
        X_coeff = array([X[0], X[1], X[2], X[0] * X[1], X[0] * X[2], X[1] * X[2],
                       X[0] ** 2, X[1] ** 2, X[2] ** 2, X[0] ** 2  * X[1], X[0] ** 2
                       * X[2], X[1] ** 2 * X[0], X[1] ** 2 * X[2], X[2] ** 2 * X[0],
                       X[2] ** 2 * X[1], X[0] ** 3, X[1] ** 3, X[2] ** 3, X[0] * X[1]
                       * X[2], ones(len)])
        x = self.calib.dot(X_coeff)
        if a == 1:
            x = x.reshape(2)
        return x - dif



def readimage(filename):
    """ Read grayscale image from file """
    im=Image.open(filename)
    s=im.tobytes()
    if im.mode=='L':        # 8 bit image
        gray=numpy.fromstring(s,numpy.uint8)/255.0
    elif im.mode=='I;16':   # 16 bit image (assume 12 bit grayscale)
        gray=numpy.fromstring(s,numpy.uint16)/4095.0
    else:
        raise ImageFormatNotSupported
    gray=numpy.reshape(gray,(im.size[1],im.size[0]))        
    return gray

def saveimage(image,filename):
    """ Save float array (values from 0 to 1) as 8 bit grayscale image """
    imwork=image.copy()
    imwork[imwork<0]=0
    imwork[imwork>1]=1
    im8bit=(255*imwork).astype(numpy.uint8)
    im=Image.fromstring('L',(image.shape[1],image.shape[0]),im8bit.tostring())
    im.save(filename)