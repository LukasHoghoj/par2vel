"""
Class to hold interrogation grid for par2vel
"""
# Copyright Knud Erik Meyer 2017
# Open source software under the terms of the GNU General Public License ver 3

import numpy
from .camera import Camera

class Field2D(object):
    """Contains coordinates, windows and results for interrogation of image"""
    # The following internal variables are used in an object:
    #   winsize:    side length of window
    #   wintype:    type of window (e.g. square)
    #   shape:      tuple with shape of coordinate grid 
    #   x:          array with images coordinates of window centers (vector)
    #   dx:         array with local displacements (vector)
    #   outlier:    boolean array marking outliers
    #   replaced:   boolean array marking replaced vectors.
    #
    #  x is an array with x(2,ni,nj) with first index indicating direction

    def __init__(self, camera):
        self.camera = camera  # The field has a camera associated

    def squarewindows(self, winsize, overlap):
        """Make square interrogation windows with overlap

           winsize is in pixels (e.g. 32 result in 32x32 pixel areas)
           overlap is a fraction (e.g. 0.5 is 50% overlap)
        """
        self.winsize=winsize
        self.wintype='square'
        # get image size
        ni ,nj = self.camera.shape
        # find number of windows (nj is x0-direction, nj is x1-direction)
        offset = (1 - overlap) * winsize
        nx = (nj - winsize) / offset + 1
        ny = (ni - winsize)/ offset + 1
        self.shape=(int(nx), int(ny))
        # find coordinates of window centers
        mgrid=numpy.mgrid
        self.x = (mgrid[0.:nx, 0.:ny]) * offset + winsize/2.0 - 0.5
        # set outlier boolean array to false
        self.outlier = numpy.zeros(self.shape, bool)
    
    def setwinsize(self,overlap):
        """Find size of squarewindows with a certain overlap, when the 
        interogation grid already is given (most likely to be used in a stereo-
        PIV application, together with Field3D"""
        self.wintype = 'square'
        self.winsize = 32 # Have to find a better way to set winsize, eventu-
        # -ally different for each point
        nx, ny = numpy.shape(self.x[0])
        self.shape = (nx,ny)

    def setx(self, x, winsize=32):
        """Set centers of interrogation windows manually"""
        self.winsize = winsize
        self.wintype = 'square'
        self.x = numpy.array(x,float)
        if len(x.shape) < 3:
            x.reshape(2,1,-1)
        self.shape = self.x.shape[1:]

    def xflat(self):
        """Get 2xn array of window centers in image coordinates"""
        return self.x.reshape((2,-1))

    def setdxflat(self, dx):
        """Insert displacement estimates in same order as given by xflat()"""
        self.dx = dx.reshape((2,) + self.shape)
        # set (or reset) outlier boolean array to false
        self.outlier = numpy.zeros(self.shape, bool)

    def getdxflat(self):
        """Get displacement estimates in same order as given by xflat()"""
        return self.dx.reshape((2,-1))

    def maxdxfilter(self, maxdx):
        """Mark all displacements larger than maxdx as outliers"""
        dxlen = numpy.sqrt(self.dx[0,:,:]**2 + self.dx[1,:,:]**2)
        self.outlier += dxlen > maxdx

    def localmean(self,i1,j1):
        """Find local mean at indices i,j  not including outliers 
           Returns arrays with meandx, stddx and no. of points n in analysis"""
        from numpy import array, mean, std
        ni, nj = self.shape
        # relavtive indexes of half the neighbour points (lower, left part)
        irel = ((-1,-1), (0,-1), (1,-1), (-1,0))
        # step through all points
        dx0=[]; dx1=[];
        for i2, j2 in irel:
            # only point pairs (neighbour and opposite neighbour) where
            # - not outside the grid
            # - not an outlier
            if (0<=i1+i2<ni and 0<=i1-i2<ni and 0<=j1+j2<nj and 0<=j1-j2<nj
                and (not self.outlier[i1+i2,j1+j2])
                and (not self.outlier[i1-i2,j1-j2])):
                dx0.append(self.dx[0,i1+i2,j1+j2])
                dx0.append(self.dx[0,i1-i2,j1-j2])
                dx1.append(self.dx[1,i1+i2,j1+j2])
                dx1.append(self.dx[1,i1-i2,j1-j2])
        if not dx0: # if no neighbour pairs found - use mean of all
            irelall=((-1,-1),(0,-1),(1,-1),(-1,0),(1,1),(0,1),(-1,1),(1,0))
            for i2,j2 in irelall:
                if (0<=i1+i2<ni and 0<=j1+j2<nj and
                    not self.outlier[i1+i2,j1+j2]):
                    dx0.append(self.dx[0,i1+i2,j1+j2])
                    dx1.append(self.dx[1,i1+i2,j1+j2])
        if dx0:
            meandx=[mean(dx0),mean(dx1)]
            stddx=[std(dx0),std(dx1)]
        else:
            meandx=[0.0,0.0]
            stddx=[10.0,10.0] # large value to indicate very uncertain guess
        n = len(dx0)
        return meandx,stddx,n

    def meanfilter(self,threshold):
        """Filter for deviation from neighbours and mark filtered as outliers
           Threshold is compared to difference from localmean normalized
           with standard deviation of neighbours
        """
        ni,nj=self.shape
        for i in range(0,ni):
            for j in range(0,nj):
                meandx,stddx,n=self.localmean(i,j)
                if stddx[0]>0 and stddx[1]>0:
                    devdx0=abs(meandx[0]-self.dx[0,i,j])/stddx[0]
                    devdx1=abs(meandx[1]-self.dx[1,i,j])/stddx[1]
                    if devdx0>threshold or devdx1>threshold:
                        self.outlier[i,j]=True

    def medianfilter2(self,threshold=2.0):
        """Filter for deviations from neighbours and marks filtered as outliers
           Threshold is compared to difference from local median normalized
           with median of deviations from the median, see Westerweel and
           Scarano (2005). The standard value of threshold is 2.0.
           This version excludes borders
        """
        from numpy import array, median, abs, sqrt, zeros
        epsilon=0.1    # expected noiselevel
        ni,nj=self.shape
        irel=array([-1, 0, 1,-1,1,-1,0,1])
        jrel=array([-1,-1,-1, 0,0, 1,1,1])
        norm=zeros(2)
        for i in range(1,ni-1):
            for j in range(1,nj-1):
                inb=i+irel
                jnb=j+jrel
                for k in range(2):
                    dxk=self.dx[k,inb,jnb]
                    mediandx=median(dxk)
                    norm[k]=abs(self.dx[k,i,j]-mediandx)/ \
                            (median(abs(dxk-mediandx))+epsilon)
                if sqrt((norm*norm).sum())>threshold:
                    self.outlier[i,j]=True

# Version that works on borders and excludes other outliers in the evaluation
    def medianfilter(self,threshold=2.0, mindeviation=1.0):
        """Filter for deviations from neighbours and marks filtered as outliers
           Threshold is compared to difference from local median normalized
           with median of deviations from the median, see Westerweel and
           Scarano (2005). The standard value of threshold is 2.0.
           This version includes borders, but excludes neighbour vectors
           that are already marked as outliers.
           Modifications compared to Westerweel and Scarano is that the
           test is made independly in each direction and that we require
           a minimum deviation, which is set to 1 pixel so that subpixel
           variations are not filtered).
        """
        from numpy import array,median,abs,sqrt
        epsilon=0.1    # expected noiselevel (also to prevent zero division)
        ni,nj=self.shape
        irel=array([-1, 0, 1,-1,1,-1,0,1])
        jrel=array([-1,-1,-1, 0,0, 1,1,1])
        for i in range(0,ni):
            for j in range(0,nj):
                inb=i+irel
                jnb=j+jrel
                neighbour=[]
                for i2,j2 in zip(inb,jnb):
                    if 0<=i2<ni and 0<=j2<nj and (not self.outlier[i2,j2]):
                        neighbour.append(self.dx[:,i2,j2])
                if len(neighbour)>4: # at least 5 neighbours for reliable test
                    neighbour=array(neighbour) # first index is neighbour no.
                    nbmedian=median(neighbour)
                    deviation=abs(self.dx[:,i,j]-nbmedian)
                    mediannbdev=median(abs(neighbour-nbmedian))
                    normdev=deviation/(mediannbdev+epsilon)
                    if (deviation[0]>mindeviation and normdev[0]>threshold) or \
                       (deviation[1]>mindeviation and normdev[1]>threshold):
                        self.outlier[i,j]=True
                    
    def replaceoutlier(self):
        """Replace outliers with local mean value"""
        from numpy import zeros,array
        # ni,nj=self.shape
        self.replaced=zeros(self.shape,bool)
        index=self.outlier.nonzero()
        for i,j in zip(index[0],index[1]):
            meandx,stddx,n=self.localmean(i,j)
            self.dx[:,i,j]=meandx
            self.replaced[i,j]=True
                         
class Field3D(object):
    """Field3D class contains all relevant objects to perform stereo PIV these 
    are:
        - one Field2D object for each camera in the system
        - X - the coordinates of the iterogation points
        - X_corners_rectangle - the coordinates of the corners of the
    interogation area
        - partial - the partial derivatives for a displacement in camera plane
    at each point
        - dX - the physical 3D displacement at each point
        
    Field3D has to be called by a line of type:
    field  = Field3D([cam1,cam2, ....])
    """
        
    def __init__(self,cam):
        """Assign one Field2D object for each camera. The field2d objects in 
        the 3D object will later be used for the cross correlations in the
        different camera planes."""
        self.field2d = []
        for i in range(len(cam)):
            self.field2d.append(Field2D(cam[i]))
        
    def corners(self, reduce_domain = [0, 0, 0, 0]):
        """Find area that both cameras cover"""
        from numpy import append, array
        # Unwrapping
        cam1 = self.field2d[0].camera
        cam2 = self.field2d[1].camera
        # Limits of each camera
        lim1 = cam1.x2X(array([[0,0,cam1.pixels[0],cam1.pixels[0]],\
                               [0,cam1.pixels[1],0,cam1.pixels[1]]]))
        lim2 = cam2.x2X(array([[0,0,cam2.pixels[0],cam2.pixels[0]],\
                               [0,cam2.pixels[1],0,cam2.pixels[1]]]))
        # Rearanging the lim arrays
        lim1 = lim1[:,lim1[0,:].argsort()]
        lim2 = lim2[:,lim2[0,:].argsort()]
        lim1[:,0:2] = lim1[:,lim1[1,0:2].argsort()]
        lim1[:,2:4] = lim1[:,lim1[1,2:4].argsort()+2]
        lim2[:,0:2] = lim2[:,lim2[1,0:2].argsort()]
        lim2[:,2:4] = lim2[:,lim2[1,2:4].argsort()+2]
        # The four corners of the common rectangle are now defined
        reduce_domain = array(reduce_domain).reshape(2,2)
        reduce_domain[:, 0] = -reduce_domain[:,0] 
        self.X_corners_rectangle = array([[max(append(lim1[0,0:2],lim2[0,0:2])),\
                                   min(append(lim1[0,2:4],lim2[0,2:4]))],\
                                  [max(append(lim1[1,[0,2]],lim2[1,[0,2]])),\
                                   min(append(lim1[1,[1,3]],lim2[1,[1,3]]))]])\
                                   - reduce_domain
        
    def grid(self, shape, reduce_domain = [0, 0, 0, 0]):
        """Make a grid that has the resolution res[0] x res[1] and make 
           corresponding camera plane grids
           The function has to be called by the """
        self.shape = numpy.array(shape)
        self.size = self.shape[0] * self.shape[1]
        # Find corners in object plane
        self.corners(reduce_domain = reduce_domain)
        # Empty matrix for object plane coordinates:
        self.X = numpy.zeros((3,self.shape[0],self.shape[1]))
        # Space between two points (in object plane)
        DeltaX = (self.X_corners_rectangle[:,1] - \
                  self.X_corners_rectangle[:,0]) / self.shape
        grid = numpy.mgrid[0:shape[0],0:shape[1]]
        self.X[0,:,:] = (grid[0,:,:] + 1/2) * DeltaX[0] + \
                         self.X_corners_rectangle[0,0]
        self.X[1,:,:] = (grid[1,:,:] + 1/2) * DeltaX[1] + \
                         self.X_corners_rectangle[1,0]
        
        # Converting to obejct plane grid coordinates to image plane 
        # coordinates
        for i in range(len(self.field2d)):
               self.field2d[i].x = self.field2d[i].camera.X2x(self.getX_flat())
               self.field2d[i].x = self.field2d[i].x.reshape(\
                                               numpy.shape(self.X[0:2]))
               self.field2d[i].setwinsize(0.5)
    
    def getX_flat(self):
        """Returns a 3D vector in the Field3D object, inspired by the func
        from Field2D"""
        return self.X.reshape((self.X.shape[0],-1))
        
    def dxdX(self):
        """Define partial derivatives dx/dX. Using numerical differentiation
        The partial derivatives are return in a 4 dimentional array:
        Field3D.partial[field2d#,object dimension (dX,dY,dZ),camera dimenstion(
        dx,dy),ni,nj]"""
        from numpy import zeros, arange, log10, array
        from numpy.linalg import norm
        # Array with indices, that will be used to decompose the array containing
        # the Jacobi matrices
        ind = arange(self.size * 3)
        # Creating zero array, that will contain the partial derivatives:
        self.partial = zeros((len(self.field2d),3,2,self.shape[0],self.shape[1]))
        for i in range(len(self.field2d)):
            # Find the numerical tolerance for the definition of the jacobian
            tol = self.field2d[i].camera.dx2dX(self.field2d[i].x[:, round(self.shape[0]/2),\
                     round(self.shape[1] / 2)].reshape(2,1), array([[1],[1]]))
            tol = -round(log10(norm(tol)/1e+4))
            # Finding the jacobi matrix for each point
            part = self.field2d[i].camera.part(self.getX_flat(), tol)
            # Redistribution of the partial derivatives, first by dX, dY and dZ
            # and then reshaping them back into the grid
            self.partial[i,0,:,:,:] = part[:, ind % 3 == 0].reshape(2,self.shape[0],self.shape[1])
            self.partial[i,1,:,:,:] = part[:, ind % 3 == 1].reshape(2,self.shape[0],self.shape[1])
            self.partial[i,2,:,:,:] = part[:, ind % 3 == 2].reshape(2,self.shape[0],self.shape[1])
            
    def cam_dis(self):
        """ This function sets up a 1D array, that contains the camera 
        displacements"""
        
        dx1_flat = self.field2d[0].getdxflat()
        dx2_flat = self.field2d[1].getdxflat()
        self.dx_both = numpy.zeros(4*self.size)
        i = numpy.indices(self.dx_both.shape)
        i = i[0]
        # Assign a 4*nx*ny vector with all displacements in the camera planes:
        self.dx_both[i%4 == 0] = dx1_flat[0]
        self.dx_both[(i-1)%4 == 0] = dx1_flat[1]
        self.dx_both[(i-2)%4 == 0] = dx2_flat[0]
        self.dx_both[(i-3)%4 == 0] = dx2_flat[1]

    def stereo(self):
        """Uses the results from each camera plane, to find the 3D object plane
        displacements."""
        from numpy.linalg import lstsq
        from numpy import zeros, mgrid
        import time
        # Check if camera displacements exist:
        for i in range(len(self.field2d)):
            try:
                self.field2d[i].dx
            except NameError:
                raise NameError('Camera displacements have to be defined under the variable dx in the field2d subobjects')
        # Call function containing partial derivatives for movement at each point
        # in the fields
        self.dxdX()
        part = zeros((4, 3))
        dx = zeros(4)
        self.dX = zeros((3 , self.shape[0] , self.shape[1]))
        t = time.time()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                part[[ 0 , 2 ] , 0] = self.partial[: , 0 , 0 , i , j]
                part[[ 1 , 3 ] , 0] = self.partial[: , 0 , 1 , i , j]
                part[[ 0 , 2 ] , 1] = self.partial[: , 1 , 0 , i , j]
                part[[ 1 , 3 ] , 1] = self.partial[: , 1 , 1 , i , j]
                part[[ 0 , 2 ] , 2] = self.partial[: , 2 , 0 , i , j]
                part[[ 1 , 3 ] , 2] = self.partial[: , 2 , 1 , i , j]
                dx[0 : 2] = self.field2d[0].dx[: , i , j]
                dx[2 : 4] = self.field2d[1].dx[: , i , j]
                self.dX[: , i , j] = lstsq(part , dx)[0]
        # print("Time to solve the equation %s seconds" % (time.time()-t))
    
