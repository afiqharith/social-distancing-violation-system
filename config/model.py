import cv2
import os 

class LoadModelFromDisk:
    '''
    Load weights and configuration file
    ===================================
    - param: **kwargs : (UTILSDIR, MODELDIR, WEIGHTS, CFG, COCONAMES, CUDA)
    1. UTILSDIR    : utils folder
    2. MODELDIR    : model folder located in utils folder
    3. WEIGHTS     : YOLOv3 weights file located in model folder
    4. CFG         : YOLOv3 config file located in model folder
    5. COCONAMES   : file of the list of the COCO object names in the dataset
    6. CUDA        : CUDA utilization for OpenCV DNN
    '''
    def __init__(self, **kwargs) -> None:
        self.WEIGHTSPATH: str = self.get_weight_path(kwargs)
        self.CFGPATH: str = self.get_config_path(kwargs)
        self.COCO_NAMEPATH: str = self.get_coconames_path(kwargs)
        self.dnn_backend_and_target: bool = self.get_dnn_backend_and_target(kwargs)

    def get_dnn_backend_and_target(self, kwargs: dict) -> bool:
        '''
        Get DNN backend and target to utilize CPU/CUDA
        ----------------------------------------------
        '''
        return kwargs['cuda']

    def get_weight_path(self, kwargs: dict) -> str:
        '''
        Get YOLO weights file path
        --------------------------
        '''
        return os.path.join(os.getcwd(), kwargs['utilsdir'], kwargs['modeldir'], kwargs['weights'])
    
    def get_config_path(self, kwargs: dict) -> str:
        '''
        Get YOLO config file path
        -------------------------
        '''
        return os.path.join(os.getcwd(), kwargs['utilsdir'], kwargs['modeldir'], kwargs['cfg'])
    
    def get_coconames_path(self, kwargs: dict) -> str:
        '''
        Get COCO object names file path
        -------------------------------
        '''
        return os.path.join(os.getcwd(), kwargs['utilsdir'], kwargs['labelsdir'], kwargs['coco'])
    
    def get_object_classes(self) -> list:
        '''
        Gett COCO object names file path
        --------------------------------
        '''
        with open(self.COCO_NAMEPATH, "r") as f:
            class_list = [line.strip() for line in f.readlines()]
        return class_list
    
    def get_network_layer(self) -> object:
        '''
        Load weights and configuration file
        -----------------------------------
        '''
        print(f"[INFO] {self.__str__()} loaded successfully")
        return cv2.dnn.readNet(self.WEIGHTSPATH, self.CFGPATH)
    
    def get_layer_names(self) -> list:
        '''
        Get list of layers name
        -----------------------
        '''
        pre_layer_names = self.network.getLayerNames()
        return [pre_layer_names[i[0] - 1] for i in self.network.getUnconnectedOutLayers()]
    
    def setup_CPU_preferable_backend_and_target(self) -> None:
        print(f"[INFO] Setting up preferable backend and target to CPU\n")
        self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    def setup_CUDA_preferable_backend_and_target(self) -> None:
        print(f"[INFO] Setting up preferable backend and target to CUDA\n")
        self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def __str__(self) -> str:
        return self.WEIGHTSPATH.split("\\")[-1].split(".")[0].upper()

class Model(LoadModelFromDisk):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        '''
        Initialize all methods during runtime
        '''
        self.classes = self.get_object_classes()
        self.network = self.get_network_layer()
        self.layer_names = self.get_layer_names()
        '''
        Initialize OpenCV DNN to utilize CPU/CUDA during the runtime
        '''
        if self.dnn_backend_and_target:
            self.setup_CUDA_preferable_backend_and_target()
        else:
            self.setup_CPU_preferable_backend_and_target()