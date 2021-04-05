import cv2
import numpy as np
import os 

class Model:

    def __init__(self, **kwargs):
        '''
        :param all: (UTILSDIR, MODELDIR, WEIGHTS, CFG, COCONAMES)
        :param UTILSDIR: utils folder
        :param MODELDIR: model folder located in utils folder
        :param WEIGHTS: YOLOv3 weights file located in model folder
        :param CFG: YOLOv3 config file located in model folder
        :param COCONAMES: file of the list of the COCO object names in the dataset
        '''
        self.WEIGHTSPATH = os.path.join(os.getcwd(), kwargs['utilsdir'], kwargs['modeldir'], kwargs['weights'])
        self.CFGPATH = os.path.join(os.getcwd(), kwargs['utilsdir'], kwargs['modeldir'], kwargs['cfg'])
        self.COCO_NAMEPATH = os.path.join(os.getcwd(), kwargs['utilsdir'], kwargs['labelsdir'], kwargs['coco'])

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