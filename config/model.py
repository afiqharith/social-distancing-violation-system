import cv2
import os 

class Model:
    '''
    Loading weights and configuration file
    --------------------------------------
    - param: kwargs : (UTILSDIR, MODELDIR, WEIGHTS, CFG, COCONAMES)
    - UTILSDIR    : utils folder
    - MODELDIR    : model folder located in utils folder
    - WEIGHTS     : YOLOv3 weights file located in model folder
    - CFG         : YOLOv3 config file located in model folder
    - COCONAMES   : file of the list of the COCO object names in the dataset
    '''
    def __init__(self, **kwargs):
        self.WEIGHTSPATH = os.path.join(os.getcwd(), kwargs['utilsdir'], kwargs['modeldir'], kwargs['weights'])
        self.CFGPATH = os.path.join(os.getcwd(), kwargs['utilsdir'], kwargs['modeldir'], kwargs['cfg'])
        self.COCO_NAMEPATH = os.path.join(os.getcwd(), kwargs['utilsdir'], kwargs['labelsdir'], kwargs['coco'])

    def predict(self):
        '''
        Prediction
        ----------
        - return value : network, network layers name, object classes
        '''
        self.classes = list()
        with open(self.COCO_NAMEPATH, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        self.net = cv2.dnn.readNet(self.WEIGHTSPATH, self.CFGPATH)
        self.layerNames = self.net.getLayerNames()
        self.layerNames = [self.layerNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print(f"[STATUS] {Model.__str__(self)} loaded successfully\n")
        return self.net, self.layerNames, self.classes

    def __str__(self):
        return self.WEIGHTSPATH.split("\\")[-1].split(".")[0].upper()