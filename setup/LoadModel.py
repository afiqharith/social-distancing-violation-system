import cv2
import numpy as np
import os 

FOLDERPATH  = "utils/model/"
WEIGHTS     = "yolov3.weights"
CFG         = "yolov3.cfg"

class LoadModel:

    def get():

        WEIGHTSPATH = os.path.join(os.getcwd(), FOLDERPATH, WEIGHTS)
        CFGPATH = os.path.join(os.getcwd(), FOLDERPATH, CFG)
        COCO_NAMEPATH = os.path.join(os.getcwd(), "utils/", "coco.names")
        
        net = cv2.dnn.readNet(WEIGHTSPATH, CFGPATH)
        classes = []
        with open(COCO_NAMEPATH, "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layer_names = net.getLayerNames()
        outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        return net, outputlayers, classes