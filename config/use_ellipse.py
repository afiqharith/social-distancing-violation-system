import cv2

def use_ellipse(*args):
    '''
    Ellipse Implementation
    ----------------------
    - param : args : (frame, xmin, ymin, xmax, ymax, color)
    - frame : continuos frame stream
    - ellipse: image, ((center_coordinates), (axesLength), angle), startAngle, endAngle, color, thickness
    '''
    axes_1 = (((args[3] + args[1])/2) + args[1])/2
    axes_2 = (((args[4] + args[2])/2) + args[2])/2
    cv2.ellipse(args[0], ((((args[3] + args[1])/2), args[4]), (axes_1,axes_2), 285), args[5], 2)
