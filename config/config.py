'''
Configuration for the program
-----------------------------
1. CAMERA_FLAG : Input as camera stream or video stream
2. CAMERA_ID : Input camera ID
3. THREAD : Activate new thread for the program
4. THRESHOLD : Non-max supression threshold
5. CONFIDENCE : Filter weak detection predection
'''
CAMERA_FLAG     = True
CAMERA_ID       = 0
THREAD          = True
THRESHOLD       = 0.4
CONFIDENCE      = 0.5
DASHBOARD_FLAG  = False

# Model path
UTILSDIR    = "utils"
MODELDIR    = "model"
WEIGHTS     = "yolov3.weights"
CFG         = "yolov3.cfg"
LABELSDIR   = "labels"
COCONAMES   = "coco.names"
DASHBOARD   = "images/chart.png"

# Video path
FOLDERNAME  = "videos"
VIDEONAME   = "TownCentre.mp4"
WIDTH       = 1280
HEIGHT      = 720
DISTANCE    = 68.5

# Colors configurations
GREEN       = (50,205,50)
RED         = (0,0,255)
YELLOW      = (0,255,255)
ORANGE      = (0,165,255)
BLUE        = (255,0,0)
GREY        = (192,192,192)
BLACK       = (0,0,0)
WHITE       = (255,255,255)

# Dashboard colors
RED_DB      = '#FF0000'
GREEN_DB    = '#13FF00'