__author__ = "Afiq Harith"
__email__ = "afiqharith05@gmail.com"
__date__ = "08 Oct 2020"
__status__ = "Development-OOP"

import cv2
import numpy as np
import math
import os
from setup.model import dataFromModel
from setup.config import *

# Load video
VIDEOPATH = os.path.join(os.getcwd(), FOLDERNAME, VIDEONAME)

class SODV:
    def __init__(self, VIDEOPATH, DISTANCE, CAMERA, START = True):

        if CAMERA == True:
            self.video = cv2.VideoCapture(0)
        else:
            self.video = cv2.VideoCapture(VIDEOPATH)    

        self.distance = DISTANCE

        if START == True:
            self.main()
    
    def calculateCentroid(self, xmn, ymn, xmx, ymx):
        return (((xmx + xmn)/2), ((ymx + ymn)/2))
    
    def calculateDistance(self, xc1, yc1, xc2, yc2):
        # Apply Euclidean distance between two centre points
        return math.sqrt((xc1-xc2)**2 + (yc1-yc2)**2)
    
    def drawDetectionBox(self, frame, xmn, ymn, xmx, ymx, color):
        cv2.rectangle(frame, (xmn, ymn), (xmx, ymx), color, 2)
    
    def main(self):
        try:
            net, layerNames, classes = dataFromModel.get(MODELPATH, WEIGHTS, CFG, COCONAMES)
            print('[PASSED] Model loaded.')
        except:
            print('[FAILED] Unable to load model.')
        
        print('[PASSED] Starting application.')
        while (self.video.isOpened()):

            HighRiskCounter, LowRiskCounter = 0, 0
            centroids = []
            boxColors = []
            detectedBox = []

            self.ret, self.frame = self.video.read() 

            # Resize frame for prediction 
            if self.ret:
                frameResized = cv2.resize(self.frame, (416, 416))       
            else:
                break

            height, width, channels = self.frame.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(frameResized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(layerNames)

            classIDs = []
            confidences = []
            boxes = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # If class is not 0 which is person, ignore it.
                    if classID != 0:
                        continue

                    # If prediction is 50% 
                    if confidence > CONFIDENCE:
                        
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # Apply non-max suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]

                    xmn = x
                    ymn = y
                    xmx = (x + w)
                    ymx = (y + h)

                    #calculate centroid point for bounding boxes
                    centroid = self.calculateCentroid(xmn, ymn, xmx, ymx)
                    detectedBox.append([xmn, ymn, xmx, ymx, centroid])

                    violation = False
                    for k in range (len(centroids)):
                        c = centroids[k]
                        
                        if self.calculateDistance(c[0], c[1], centroid[0], centroid[1]) <= self.distance:
                            boxColors[k] = True
                            violation = True
                            cv2.line(self.frame, (int(c[0]), int(c[1])), (int(centroid[0]), int(centroid[1])), YELLOW, 1, cv2.LINE_AA)
                            cv2.circle(self.frame, (int(c[0]), int(c[1])), 3, ORANGE, -1,cv2.LINE_AA)
                            cv2.circle(self.frame, (int(centroid[0]), int(centroid[1])), 3, ORANGE, -1, cv2.LINE_AA)
                            break
                    centroids.append(centroid)
                    boxColors.append(violation)        

            for i in range (len(detectedBox)):
                xmin = detectedBox[i][0]
                ymin = detectedBox[i][1]
                xmax = detectedBox[i][2]
                ymax = detectedBox[i][3]
                
                if boxColors[i] == False:
                    self.drawDetectionBox(self.frame, xmin, ymin, xmax, ymax, WHITE)
                    label = "LOW-risk"
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    ylabel = max(ymin, labelSize[1])
                    cv2.rectangle(self.frame, (xmin, ylabel - labelSize[1]), (xmin + labelSize[0], ymin + baseLine), WHITE, cv2.FILLED)
                    cv2.putText(self.frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
                    LowRiskCounter += 1

                else:
                    self.drawDetectionBox(self.frame, xmin, ymin, xmax, ymax, RED)
                    label = "HIGH-risk"
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    ylabel = max(ymin, labelSize[1])
                    cv2.rectangle(self.frame, (xmin, ylabel - labelSize[1]),(xmin + labelSize[0], ymin + baseLine), WHITE, cv2.FILLED)
                    cv2.putText(self.frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1, cv2.LINE_AA)
                    HighRiskCounter += 1

            cv2.rectangle(self.frame, (13, 10),(250, 60), GREY, cv2.FILLED)
            LINE = "--"
            HIGHRISK_TEXT = f'HIGH RISK: {HighRiskCounter} people'
            cv2.putText(self.frame, LINE, (28, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2, cv2.LINE_AA)
            cv2.putText(self.frame, HIGHRISK_TEXT, (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1, cv2.LINE_AA)

            LOWRISK_TEXT = f'LOW  RISK: {LowRiskCounter} people'
            cv2.putText(self.frame, LINE, (28, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2, cv2.LINE_AA)
            cv2.putText(self.frame, LOWRISK_TEXT, (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1, cv2.LINE_AA)

            cv2.imshow("SODV", self.frame)

            if cv2.waitKey(1) >= 0:  
                break

        self.video.release()

if __name__ == '__main__':
    SODV(VIDEOPATH, DISTANCE, CAMERA)
    cv2.destroyAllWindows()