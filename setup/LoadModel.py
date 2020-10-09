import cv2
import numpy as np
import os 

class LoadModel:

    def get():

        foldersPath = "utils/model/"
        weightsPath = os.path.join(os.getcwd(), foldersPath, "yolov3.weights")
        cfgPath = os.path.join(os.getcwd(), foldersPath, "yolov3.cfg")
        coco_namePath = os.path.join(os.getcwd(), "utils/", "coco.names")
        
        net = cv2.dnn.readNet(weightsPath, cfgPath)
        classes = []
        with open(coco_namePath, "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layer_names = net.getLayerNames()
        outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        return net, outputlayers, classes