import json
import os

class Config:
    '''
    curation for the program
    =========================
    1. CAMERA_FLAG : Input as camera stream or video stream
    2. CAMERA_ID : Input camera ID
    3. THREAD : Activate new thread for the program
    4. THRESHOLD : Non-max supression threshold
    5. CONFIDENCE : Filter weak detection predection
    '''
    def __init__(self):
        self.json_path = os.path.join(os.getcwd(), 'config', 'config.json')
        with open (self.json_path, 'r') as fileIn:
            config = json.load(fileIn)
            file = config.get("file")
            attributes = config.get("attributes")
            color = config.get("color")

        self.UTILSDIR        = file['UTILSDIR']
        self.MODELDIR        = file['MODELDIR']
        self.WEIGHTS         = file['WEIGHTS']
        self.CFG             = file['CFG']
        self.LABELSDIR       = file['LABELSDIR']
        self.COCONAMES       = file['COCONAMES']
        self.DASHBOARD       = file['DASHBOARD']
        self.FOLDERNAME      = file['FOLDERNAME']
        self.VIDEONAME       = file['VIDEONAME']

        self.CAMERA_FLAG     = attributes['CAMERA_FLAG']
        self.CAMERA_ID       = attributes['CAMERA_ID']
        self.THREAD          = attributes['THREAD']
        self.THRESHOLD       = attributes['THRESHOLD']
        self.CONFIDENCE      = attributes['CONFIDENCE']
        self.DASHBOARD_FLAG  = attributes['DASHBOARD_FLAG']
        self.LOGGING         = attributes['LOGGING']
        self.WIDTH           = attributes['WIDTH']
        self.HEIGHT          = attributes['HEIGHT']
        self.DISTANCE        = attributes['DISTANCE']

        self.GREEN           = color["bgr"]['GREEN']
        self.RED             = color["bgr"]['RED']
        self.YELLOW          = color["bgr"]['YELLOW']
        self.ORANGE          = color["bgr"]['ORANGE']
        self.BLUE            = color["bgr"]['BLUE']
        self.GREY            = color["bgr"]['GREY']
        self.BLACK           = color["bgr"]['BLACK']
        self.WHITE           = color["bgr"]['WHITE']
        self.RED_DB          = color["hex"]['RED_DB']
        self.GREEN_DB        = color["hex"]['GREEN_DB']

    def __str__(self) -> str:
        return