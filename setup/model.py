import cv2
import numpy as np
import os 

class Model:

    def __init__(self, UTILS, MODELPATH, WEIGHTS, CFG, COCONAMES):
        '''
        :param UTILS: utils folder
        :param MODELPATH: model folder located in utils folder
        :param WEIGHTS: YOLOv3 weights file located in model folder
        :param CFG: YOLOv3 config file located in model folder
        :param COCONAMES: file of the list of the COCO object names in the dataset
        '''
        self.WEIGHTSPATH = os.path.join(os.getcwd(), UTILS, MODELPATH, WEIGHTS)
        self.CFGPATH = os.path.join(os.getcwd(), UTILS, MODELPATH, CFG)
        self.COCO_NAMEPATH = os.path.join(os.getcwd(), UTILS, COCONAMES)

    def predict(self):

        classes = list()
        with open(self.COCO_NAMEPATH, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        
        net = cv2.dnn.readNet(self.WEIGHTSPATH, self.CFGPATH)
        layerNames = net.getLayerNames()
        layerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        return net, layerNames, classes