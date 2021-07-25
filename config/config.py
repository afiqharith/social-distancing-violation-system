import json
import os

class LoadDataFromJson:
    json_path = os.path.join(os.getcwd(), 'config', 'config.json')
    with open (json_path, 'r') as fileIn:
        config = json.load(fileIn)
        file = config.get("file")
        attributes = config.get("attributes")
        color = config.get("color")

class Path(LoadDataFromJson):
    '''
    Path variables for the program
    '''
    UTILSDIR: str = LoadDataFromJson().file['UTILSDIR']
    MODELDIR: str = LoadDataFromJson().file['MODELDIR']
    WEIGHTS: str = LoadDataFromJson().file['WEIGHTS']
    CFG: str = LoadDataFromJson().file['CFG']
    LABELSDIR: str = LoadDataFromJson().file['LABELSDIR']
    COCONAMES: str = LoadDataFromJson().file['COCONAMES']
    DASHBOARD_PATH: str = LoadDataFromJson().file['DASHBOARD_PATH']
    FOLDERNAME: str = LoadDataFromJson().file['FOLDERNAME']
    VIDEONAME: str = LoadDataFromJson().file['VIDEONAME']

class Attributes(LoadDataFromJson):
    '''
    Attribute variables for the program
    '''
    CAMERA_FLAG: bool = LoadDataFromJson().attributes['CAMERA_FLAG']
    CAMERA_ID: int = LoadDataFromJson().attributes['CAMERA_ID']
    THREAD: bool = LoadDataFromJson().attributes['THREAD']
    NMS_THRESHOLD: float = LoadDataFromJson().attributes['NMS_THRESHOLD']
    MIN_CONFIDENCE: float = LoadDataFromJson().attributes['MIN_CONFIDENCE']
    DASHBOARD_FLAG: bool = LoadDataFromJson().attributes['DASHBOARD_FLAG']
    LOGGING: bool = LoadDataFromJson().attributes['LOGGING']
    WIDTH: int = LoadDataFromJson().attributes['WIDTH']
    HEIGHT: int = LoadDataFromJson().attributes['HEIGHT']
    DISTANCE: float = LoadDataFromJson().attributes['DISTANCE']

class Colors(LoadDataFromJson):
    '''
    Color variables for the program
    '''
    GREEN: list= LoadDataFromJson().color["bgr"]['GREEN']
    RED: list = LoadDataFromJson().color["bgr"]['RED']
    YELLOW: list = LoadDataFromJson().color["bgr"]['YELLOW']
    ORANGE: list = LoadDataFromJson().color["bgr"]['ORANGE']
    BLUE: list = LoadDataFromJson().color["bgr"]['BLUE']
    GREY: list = LoadDataFromJson().color["bgr"]['GREY']
    BLACK: list = LoadDataFromJson().color["bgr"]['BLACK']
    WHITE: list = LoadDataFromJson().color["bgr"]['WHITE']
    RED_DB: list = LoadDataFromJson().color["hex"]['RED_DB']
    GREEN_DB: list = LoadDataFromJson().color["hex"]['GREEN_DB']

class Config(Path, Attributes, Colors):
    def __init__(self) -> None:
        super().__init__()