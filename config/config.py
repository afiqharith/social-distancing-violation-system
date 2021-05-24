import json
import os

CONFIGJSON = os.path.join(os.getcwd(), 'config', 'config.json')
with open (CONFIGJSON) as fileIn:
    config = json.load(fileIn)
    file = config.get("file")
    attributes = config.get("attributes")
    color = config.get("color")

'''
curation for the program
-----------------------------
1. CAMERA_FLAG : Input as camera stream or video stream
2. CAMERA_ID : Input camera ID
3. THREAD : Activate new thread for the program
4. THRESHOLD : Non-max supression threshold
5. CONFIDENCE : Filter weak detection predection
'''

UTILSDIR    = file['UTILSDIR']
MODELDIR    = file['MODELDIR']
WEIGHTS     = file['WEIGHTS']
CFG         = file['CFG']
LABELSDIR   = file['LABELSDIR']
COCONAMES   = file['COCONAMES']
DASHBOARD   = file['DASHBOARD']
FOLDERNAME  = file['FOLDERNAME']
VIDEONAME   = file['VIDEONAME']

CAMERA_FLAG     = attributes['CAMERA_FLAG']
CAMERA_ID       = attributes['CAMERA_ID']
THREAD          = attributes['THREAD']
THRESHOLD       = attributes['THRESHOLD']
CONFIDENCE      = attributes['CONFIDENCE']
DASHBOARD_FLAG  = attributes['DASHBOARD_FLAG']
LOGGING         = attributes['LOGGING']
WIDTH           = attributes['WIDTH']
HEIGHT          = attributes['HEIGHT']
DISTANCE        = attributes['DISTANCE']

GREEN           = color["bgr"]['GREEN']
RED             = color["bgr"]['RED']
YELLOW          = color["bgr"]['YELLOW']
ORANGE          = color["bgr"]['ORANGE']
BLUE            = color["bgr"]['BLUE']
GREY            = color["bgr"]['GREY']
BLACK           = color["bgr"]['BLACK']
WHITE           = color["bgr"]['WHITE']
RED_DB          = color["hex"]['RED_DB']
GREEN_DB        = color["hex"]['GREEN_DB']