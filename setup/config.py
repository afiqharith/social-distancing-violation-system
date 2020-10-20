# App configuration
CAMERA = False

# Threshold when apply NMS 
THRESHOLD = 0.4

# Detection confidence to filter weak prediction
CONFIDENCE = 0.5

# Model path
MODELPATH = "utils/model/"
WEIGHTS = "yolov3.weights"
CFG = "yolov3.cfg"
COCONAMES = "coco.names"

# Video path
FOLDERNAME = "videos/"
VIDEONAME = "TownCentre.mp4"
WIDTH = 1280
HEIGHT = 720
DISTANCE = 68.5

'''
# PETS2009
FOLDERNAME = "videos/"
VIDEONAME = "PETS2009.mp4"
WIDTH = 1280
HEIGHT = 720
DISTANCE = 70

# VIRAT
FOLDERNAME = "videos/"
VIDEONAME = "VIRAT.mp4"
WIDTH = 1280
HEIGHT = 720
DISTANCE = 55
'''

# Colors configuration
GREEN = (50,205,50)
RED = (0,0,255)
YELLOW = (0,255,255)
WHITE = (255,255,255)
ORANGE = (0,165,255)
BLUE = (255,0,0)
GREY = (192,192,192)