class Calibration_image(object):
    def __init__(self, img_name):
        """The Calibration_image is called as Calibration_image(img_name), 
        where img_name is the path to the calibration image. 
        The purpose of this object is to read an image of a carthesian grid, 
        where each point is represented by a dot. Call the object in order to 
        open an image.
        It may be necessary to do some preleminary picture processing in order
        to eliminate shadows.
        
        >>> self = Calibration_image(filename)
        """

        from cv2 import imread

        self.img = imread(img_name,0)
        self.shape = self.img.shape
        self.img_name = img_name
    
    def find_ellipses(self, edge_dist = [15, 15, 15, 15]):
        """The find_ellipses function detects elipse shapes, that correspond to
        dots, in the image. The center coordinates of all elipses, which lie in
        the acceptebale windows are returned in the array self.x. 
        The optional argument edge_dist can be used to determine how close to 
        the edge the centerpoints may lie. The distances are defined in pixels
        on the image plane
        
        >>> self.find_ellipses(edge_dist = [d_top, d_bottom, d_left, d_right])
        """

        from cv2 import Canny, findContours, fitEllipse, RETR_EXTERNAL, \
                        CHAIN_APPROX_NONE
        from numpy import zeros

        # If edge list is only a scalar, use the distance on all sides
        if len(edge_dist) == 1:
            edge_dist = [edge_dist] * 4
        # Empty lists
        elpsa = []
        x = []
        y = []
        # Canny image processing, finds pixels with big contrast
        thresh = Canny(self.img, 80, 255)
        # Find contours
        b, contours, hierarchy = findContours(thresh, RETR_EXTERNAL,\
                                                  CHAIN_APPROX_NONE)
        for ind, cont in enumerate(contours):
            elpsa.append(fitEllipse(cont))
            # If the centerpoint is not to close to the edge
            if elpsa[ind][0][0] > edge_dist[2] and\
               elpsa[ind][0][0] < self.shape[1] - edge_dist[3] and\
               elpsa[ind][0][1] > edge_dist[0] and\
               elpsa[ind][0][1] < self.shape[0] - edge_dist[1]:
                # Add centerpoint to previoulsy created list
                x.append(elpsa[ind][0][0])
                y.append(elpsa[ind][0][1])
        
        # Store in numpy array
        self.x = zeros((2, len(x)))
        self.x[:, :] = [x, y]
        

    def coordinate_system(self):
        """This function will display the image with a red cross at each 
        dot centerpoint. The function will let the user click on the 
        centerpoitn as well as the points respectively defining the x and
        y axis. 

        >>> self.coordinate_system()
        """

        from matplotlib.pyplot import figure, imshow, title, plot, show, \
                                      pause, subplot2grid, text, axis
        from numpy import array

        self.selected_x = []
        self.selected_y = []
        self.quit_figure = None
        self.fig = figure(1)
        self.ax_im = subplot2grid((10, 3), (0,0), colspan = 3, rowspan = 9)
        imshow(self.img,'gray')
        title(("Please click on the following points in the same order as"+\
               "mentioned: (0,0), (1,0), (0,1)\n " +\
               "Then click on points that have to be exclude from the"+\
               " axes, such that all defined points\n" +\
               "on the axes belong to complete lines of dots"), fontsize = 8)
        plot(self.x[0, :],self.x[1, :],'rx')
        axis('off')
        self.ax_done = subplot2grid((10, 3), (9,0))
        text(0.5, 0.5,'Done',horizontalalignment='center',\
                             verticalalignment='center')
        axis([0, 1, 0, 1])
        axis('off')
        self.ax_close = subplot2grid((10, 3), (9, 1))
        text(0.5, 0.5, 'Done and close figure', horizontalalignment = 'center'\
                        , verticalalignment='center')
        axis([0, 1, 0, 1])
        axis('off')
        self.ax_del = subplot2grid((10, 3), (9, 2))
        text(0.5, 0.5, 'Cancel last selected point',\
                 horizontalalignment = 'center', verticalalignment = 'center')
        axis([0, 1, 0, 1])
        axis('off')
        show()
        pause(0.01)
        self.cid = self.fig.canvas.mpl_connect('button_press_event',\
                                               self.onclick)
        # Wait for the user to select 3 points:
        self.waitcanvas()
        # Delete ellipses the user does not wants to keep:
        x_del = array([self.selected_x[3:], self.selected_y[3:]])
        self.delete_ellipses(x_del)

    def onclick(self, event):
        """This function appends the x and y values of the click in the figure
        to respectively the self.selected_x and self.selected_y list.
        Note that this function already is called in both the
        self.coordinate_system and the self.waitcanvas functions.
        """
        
        if event.inaxes == self.ax_im:
            self.selected_x.append(event.xdata)
            self.selected_y.append(event.ydata)
        elif event.inaxes == self.ax_done:
            self.quit_figure = 'done'
        elif event.inaxes == self.ax_close:
            self.quit_figure = 'close'
        elif event.inaxes == self.ax_del:
            self.selected_x.pop()
            self.selected_y.pop()
        
    def waitcanvas(self):
        """The main purpose of this function is to make the program wait 
        before executing the rest of the scripts, such that the user has time
        enough to make the 3 clicks.
        Note that this function is automatically called in 
        self.coordinate_system()
        """

        from matplotlib.pyplot import pause, close

        while True:
            pause(0.01)
            self.cid = self.fig.canvas.mpl_connect('button_press_event',\
                                                   self.onclick)
            if self.quit_figure == 'done' and len(self.selected_x) >= 3:
                break
            if self.quit_figure == 'close' and len(self.selected_x) >= 3:
                close(self.fig)
                break
            self.quit_figure = None

        self.fig.canvas.mpl_disconnect(self.cid)
    
    def closest_point(self, x_):
        """This function finds the a closest ellipse centerpoint to a 
        point [sx, sy].
        
        >>> self.closest_point(x_)
        """

        diff = ((self.x[0] - x_[0]) ** 2 + (self.x[1] - x_[1]) ** 2) ** 0.5
        p = self.x[0 : 2, diff == min(diff)]
        return p

    def delete_ellipses(self, x_del):
        """This function takes ellipse center points as input and deletes
        them from the array containing the ellipse centerpoints, such that
        those centerpoints will not be included in further processing"""
        
        from numpy import delete, where
        ind = []
        # Find indices of points, which have to be deleted
        for i in range(len(x_del[0])):
            x_del_local = self.closest_point(x_del[:, i])
            ind.append(int(where((x_del_local[0, 0] == self.x[0]) * \
                                 (x_del_local[1, 0] == self.x[1]))[0][0]))
        # Delete points:
        self.x = delete(self.x, ind, 1)

    def center_axis(self):
        """This function interpretes the user click input on the figure and
        fits the click coordinates to ellipse centerpoints. The data is then
        stored in the self.axes array, which contains the coordinates of the
        three points in object space as well as the coordinates of the center
        of the ellipse fitting the dot.
        
        >>> self.center_axis()
        """

        from numpy import array, vstack, hstack

        self.center = self.closest_point(array([self.selected_x[0],\
                                                self.selected_y[0]]))
        x_axis = self.closest_point(array([self.selected_x[1],\
                                           self.selected_y[0]]))
        y_axis = self.closest_point(array([self.selected_x[2],\
                                           self.selected_y[2]]))

        self.axes = vstack((array([[0, 1, 0], [0, 0, 1]]),\
                 hstack((self.center, x_axis, y_axis))))

    def object_coordinates(self, X_spacing = [1, 1], pixeltol = 20):
        """When the ellipse centers to be used and the points defining the axes
        are defined, the object coordinates corresponding to the other ellipse
        centerpoints can now be defined. The function takes two optional input
        arguments:
        - X_spacing, is a list or float, that defines the space between two 
        dots in physical space
        - Pixeltol, how close two points have to be to each other, before it 
        can be assumed that it is the same point.

        >>> self.object_coordinates(X_spacing = opt., pixeltol = opt.)
        """

        from numpy import vstack, ones, isclose, where, arange, sort, zeros,\
                            array
        from numpy.linalg import lstsq

        # If the spacing argument is only given as a single value, assume
        # it counts for both axes
        if len(X_spacing) == 1:
            X_spacing = [X_spacing] * 2

        # Defining the lines that respectively go through the x and y axis:
        rhs_x = vstack((self.axes[2, 0:2], ones(2))).T
        rhs_y = vstack((self.axes[2, [0, 2]], ones(2))).T
        lhs_x = self.axes[3, 0:2]
        lhs_y = self.axes[3, [0, 2]]
        a_x, b_x = lstsq(rhs_x, lhs_x)[0]
        a_y, b_y = lstsq(rhs_y, lhs_y)[0]
        # y positions of the ellipse centerpoints according to their x
        # coordinates and the linear regressiion:
        test_on_x = a_x * self.x[0] + b_x
        test_on_y = a_y * self.x[0] + b_y
        # Extract points that are close enough to axis:
        x_axis = self.x[0:2, isclose(test_on_x, self.x[1],\
                                     atol = pixeltol, rtol = 0)]
        y_axis = self.x[0:2, isclose(test_on_y, self.x[1],\
                                     atol = pixeltol, rtol = 0)]
        if self.axes[2, 0] < self.axes[2,1]:
            x_axis = x_axis[:, x_axis[0].argsort()]
        else:
            x_axis = x_axis[:, -x_axis[0].argsort()]
        if self.axes[3, 0] < self.axes[3,2]:
            y_axis = y_axis[:, (-y_axis[1]).argsort()]
        else:
            y_axis = y_axis[:, (y_axis[1]).argsort()]
        ind_center = [int(where((x_axis[0] == self.center[0]) * \
                 (x_axis[1] == self.center[1]))[0][0]),\
                int(where((y_axis[0] == self.center[0]) \
                 * (y_axis[1] == self.center[1]))[0][0])]
        X_axis = vstack((arange(len(x_axis[0])) - \
                 where((x_axis[0] == self.center[0]) * \
                 (x_axis[1] == self.center[1]))[0][0], zeros(len(x_axis[0]))))
        Y_axis = vstack((zeros(len(y_axis[0])), -arange(len(y_axis[0])) + \
                 where((y_axis[0] == self.center[0]) *\
                       (y_axis[1] == self.center[1]))[0][0]))
        # Make a grid form to simplify iteration indexing:
        X_grid = zeros((4, len(Y_axis[0]), len(X_axis[0])))
        X_grid[0:2, ind_center[1], :] = X_axis
        X_grid[2:4, ind_center[1], :] = x_axis
        X_grid[0:2, :, ind_center[0]] = Y_axis
        X_grid[2:4, :, ind_center[0]] = y_axis
        # The points along the x and y axis are now defined, the rest can now
        # be filled out:
        """
        The image is divided in four quadrants, that are processed separately.
        Gived the four points:
        x_2|x_4
        ---+---
        x_1|x_3
        Where the coordinates for x_4 have to be found, a guess value can be 
        found by vector aditions: x_4_guess = x_2 + x_3 - x_1
        The self.closest_point function is then used to relate the guess 
        coordinates to an ellipse centerpoint.
        """
        for i in X_axis[0, X_axis[0] > 0]:
            i = int(i)
            for j in sort(Y_axis[1, Y_axis[1] > 0]):
                j = int(j)
                X_grid[0:2, ind_center[1] - j, ind_center[0] + i] =\
                                                                  array([i, j])
                x_point = X_grid[2:4, ind_center[1] - j + 1,\
                                      ind_center[0] + i] +\
                          X_grid[2:4, ind_center[1] - j ,\
                                      ind_center[0] + i - 1] - \
                          X_grid[2:4, ind_center[1] - j + 1,\
                                      ind_center[0] + i - 1]
                X_grid[2:4, ind_center[1] - j, ind_center[0] + i] =\
                                              self.closest_point(x_point).T[0]
            for j in Y_axis[1, Y_axis[1] < 0]:
                j = int(j)
                X_grid[0:2, ind_center[1] - j, ind_center[0] + i] =\
                                                                  array([i, j])
                x_point = X_grid[2:4, ind_center[1] - j - 1,\
                                      ind_center[0] + i] + \
                          X_grid[2:4, ind_center[1] - j ,\
                                      ind_center[0] + i - 1] - \
                          X_grid[2:4, ind_center[1] - j - 1,\
                                      ind_center[0] + i - 1]
                X_grid[2:4, ind_center[1] - j, ind_center[0] + i] =\
                                              self.closest_point(x_point).T[0]

        for i in X_axis[0, X_axis[0] < 0][::-1]:
            i = int(i)
            for j in sort(Y_axis[1, Y_axis[1] > 0]):
                j = int(j)
                X_grid[0:2, ind_center[1] - j, ind_center[0] + i] =\
                                                                  array([i, j])
                x_point = X_grid[2:4, ind_center[1] - j + 1,\
                                      ind_center[0] + i] + \
                          X_grid[2:4, ind_center[1] - j ,\
                                      ind_center[0] + i + 1] - \
                          X_grid[2:4, ind_center[1] - j + 1,\
                                      ind_center[0] + i + 1]
                X_grid[2:4, ind_center[1] - j, ind_center[0] + i] =\
                                              self.closest_point(x_point).T[0]

            for j in Y_axis[1, Y_axis[1] < 0]:
                j = int(j)
                X_grid[0:2, ind_center[1] - j, ind_center[0] + i] =\
                                                                  array([i, j])
                x_point = X_grid[2:4, ind_center[1] - j - 1,\
                                      ind_center[0] + i] + \
                          X_grid[2:4, ind_center[1] - j ,\
                                      ind_center[0] + i + 1] - \
                          X_grid[2:4, ind_center[1] - j - 1,\
                                      ind_center[0] + i + 1]
                X_grid[2:4, ind_center[1] - j, ind_center[0] + i] =\
                                              self.closest_point(x_point).T[0]
        
        # Save the grid in the object:
        self.X_grid = X_grid

        # Scale physical coordinates:
        self.X_grid[0, :, :] = self.X_grid[0, :, :] * X_spacing[0]
        self.X_grid[1, :, :] = self.X_grid[1, :, :] * X_spacing[1]
        self.X = X_grid.reshape(4, X_grid.shape[1] * X_grid.shape[2])

    def save2file(self, Z, filename, description = None):
        """Save data in a file"""
        f = open(filename, 'w')
        f.write('# File containing coordinates for camera calibration \n')
        if description != None:
            print(description, file = f)
        f.write("Z = {:}\n".format(repr(Z)))
        for row in self.X:
            for value in row:
                print(repr(value), end = ' ', file = f)
            print(file = f)
        
    def append2file(self, Z, filename):
        """Append data to an already existing file"""
        f = open(filename, 'a')
        f.write("Z = {:}\n".format(repr(Z)))
        for row in self.X:
            for value in row:
                print(repr(value), end = ' ', file = f)
            print(file = f)
