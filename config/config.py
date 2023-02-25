import json
import os

class LoadDataFromJson:
    json_path = os.path.join(os.getcwd(), 'config', 'config.json')
    with open (json_path, 'r') as fileIn:
        config = json.load(fileIn)
        file = config.get("file")
        attributes = config.get("attributes")
        color = config.get("color")

conf = LoadDataFromJson()

class Path:
    '''
    Path variables for the program
    '''
    UTILSDIR            = conf.file['UTILSDIR']
    MODELDIR            = conf.file['MODELDIR']
    WEIGHTS             = conf.file['WEIGHTS']
    CFG                 = conf.file['CFG']
    LABELSDIR           = conf.file['LABELSDIR']
    COCONAMES           = conf.file['COCONAMES']
    DASHBOARD_FILENAME  = conf.file['DASHBOARD_FILENAME']
    FOLDERNAME          = conf.file['FOLDERNAME']
    VIDEONAME           = conf.file['VIDEONAME']
    TEMP_DIR            = conf.file['TEMP_DIR']

    CFG_ABS_PATH        = os.path.join(os.getcwd(), UTILSDIR, MODELDIR, CFG)
    WEIGHT_ABS_PATH     = os.path.join(os.getcwd(), UTILSDIR, MODELDIR, WEIGHTS)
    LABELS_ABS_PATH     = os.path.join(os.getcwd(), UTILSDIR, LABELSDIR, COCONAMES)
    DASHBOARD_PATH      = os.path.join(os.getcwd(), TEMP_DIR, DASHBOARD_FILENAME)
    LOG_PATH            = os.path.join(os.getcwd(), TEMP_DIR, "log.txt")

class Attributes:
    '''
    Attribute variables for the program
    '''
    CAMERA_FLAG: bool       = conf.attributes['CAMERA_FLAG']
    CAMERA_ID: int          = conf.attributes['CAMERA_ID']
    THREAD: bool            = conf.attributes['THREAD']
    NMS_THRESHOLD: float    = conf.attributes['NMS_THRESHOLD']
    MIN_CONFIDENCE: float   = conf.attributes['MIN_CONFIDENCE']
    DASHBOARD_FLAG: bool    = conf.attributes['DASHBOARD_FLAG']
    LOGGING: bool           = conf.attributes['LOGGING']
    WIDTH: int              = conf.attributes['WIDTH']
    HEIGHT: int             = conf.attributes['HEIGHT']
    DISTANCE: float         = conf.attributes['DISTANCE']

class Colors:
    '''
    Color variables for the program
    '''
    GREEN: list     = conf.color["bgr"]['GREEN']
    RED: list       = conf.color["bgr"]['RED']
    YELLOW: list    = conf.color["bgr"]['YELLOW']
    ORANGE: list    = conf.color["bgr"]['ORANGE']
    BLUE: list      = conf.color["bgr"]['BLUE']
    GREY: list      = conf.color["bgr"]['GREY']
    BLACK: list     = conf.color["bgr"]['BLACK']
    WHITE: list     = conf.color["bgr"]['WHITE']
    RED_DB: list    = conf.color["hex"]['RED_DB']
    GREEN_DB: list  = conf.color["hex"]['GREEN_DB']

class Config(Path, Attributes, Colors):
    def __init__(self) -> None:
        super().__init__()

if __name__ == "__main__":
    print(Attributes.CAMERA_FLAG)