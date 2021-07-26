import cv2

def use_ellipse(*axis):
    '''
    Ellipse Implementation
    ----------------------
    - param : *axis : (frame, xmin, ymin, xmax, ymax, color)
    - frame : continuos frame stream
    - ellipse: image, ((center_coordinates), (axesLength), angle), startAngle, endAngle, color, thickness
    '''
    axes_1 = (((axis[3] + axis[1])/2) + axis[1])/2
    axes_2 = (((axis[4] + axis[2])/2) + axis[2])/2
    cv2.ellipse(axis[0], ((((axis[3] + axis[1])/2), axis[4]), (axes_1,axes_2), 285), axis[5], 2)
