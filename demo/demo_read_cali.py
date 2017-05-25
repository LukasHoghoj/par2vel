from par2vel.read_calibration import Calibration_image

# Import image into object:
a = Calibration_image('calitest.jpg')
# Find ellipse centerpoints
a.find_ellipses(edge_dist = [0])
# User defined coordinate system and points that aren't taken into account
a.coordinate_system()
# Compute center axes
a.center_axis()
# Find corresponding object coordinates
a.object_coordinates()
# Save grid to global variable for visualization
b = a.X_grid
# Save to file under one Z coordinate
a.save2file(0, 'hej')
# Save to file under annother Z coordinate
a.append2file(1, 'hej')