import cv2
import os 

class Model:
    '''
    Loading weights and configuration file
    ======================================
    - param: kwargs : (UTILSDIR, MODELDIR, WEIGHTS, CFG, COCONAMES)
    1. UTILSDIR    : utils folder
    2. MODELDIR    : model folder located in utils folder
    3. WEIGHTS     : YOLOv3 weights file located in model folder
    4. CFG         : YOLOv3 config file located in model folder
    5. COCONAMES   : file of the list of the COCO object names in the dataset
    '''
    def __init__(self, **kwargs):
        self.WEIGHTSPATH = self.get_weight_path(kwargs)
        self.CFGPATH = self.get_config_path(kwargs)
        self.COCO_NAMEPATH = self.get_coconames_path(kwargs)

        self.classes = self.get_object_classes()
        self.network = self.get_network_layer()
        self.layer_names = self.get_layer_names()

        self.setup_dnn_backend()

    def get_weight_path(self, kwargs: dict) -> str:
        '''
        Getting YOLO weights file path
        ------------------------------
        '''
        return os.path.join(os.getcwd(), kwargs['utilsdir'], kwargs['modeldir'], kwargs['weights'])
    
    def get_config_path(self, kwargs: dict) -> str:
        '''
        Getting YOLO config file path
        ------------------------------
        '''
        return os.path.join(os.getcwd(), kwargs['utilsdir'], kwargs['modeldir'], kwargs['cfg'])
    
    def get_coconames_path(self, kwargs: dict) -> str:
        '''
        Getting COCO object names file path
        -----------------------------------
        '''
        return os.path.join(os.getcwd(), kwargs['utilsdir'], kwargs['labelsdir'], kwargs['coco'])
    
    def get_object_classes(self) -> list:
        '''
        Getting COCO object names file path
        -----------------------------------
        '''
        with open(self.COCO_NAMEPATH, "r") as f:
            class_list = [line.strip() for line in f.readlines()]
        return class_list
    
    def get_network_layer(self) -> object:
        '''
        Loading weights and configuration file
        --------------------------------------
        '''
        print(f"[STATUS] {Model.__str__(self)} loaded successfully\n")
        return cv2.dnn.readNet(self.WEIGHTSPATH, self.CFGPATH)
    
    def get_layer_names(self) -> list:
        '''
        Getting list of layers name
        ----------------------------
        '''
        pre_layer_names = self.network.getLayerNames()
        return [pre_layer_names[i[0] - 1] for i in self.network.getUnconnectedOutLayers()]
    
    def setup_dnn_backend(self):
        print(f"[STATUS] Setting up DNN to target CPU\n")
        self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def __str__(self) -> str:
        return self.WEIGHTSPATH.split("\\")[-1].split(".")[0].upper()