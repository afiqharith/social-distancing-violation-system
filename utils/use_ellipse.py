import cv2

def use_ellipse(frame, xmin, ymin, xmax, ymax, color):
    '''
    Ellipse Implementation
    ----------------------
    - param : *axis : (frame, xmin, ymin, xmax, ymax, color)
    - frame : continuos frame stream
    - ellipse: image, ((center_coordinates), (axesLength), angle), startAngle, endAngle, color, thickness
    '''
    #horizontal_axes = (((xmax + xmin)/2) + xmin)/2
    #vertical_axes  = (((ymax + ymin)/2) + ymin)/2
    horizontal_axes = ((xmax - xmin)/2)
    vertical_axes  = ((ymax - ymin)/2) 
    centerx = (xmax + xmin)/2
    #cv2.ellipse(frame, ((((xmax + xmin)/2), ymax), (axes_1,axes_2), 285), color, 2)
    cv2.ellipse(frame, ((centerx, ymax), (horizontal_axes, vertical_axes), 350), color, 2)
