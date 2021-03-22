#!/Users/Afiq/Envs/global/Scripts/python
__author__ = "Afiq Harith"
__email__ = "afiqharith05@gmail.com"
__date__ = "08 Oct 2020"
__status__ = "Development"

from setup.model import Model
from setup.config import *
from time import time
import numpy as np
import math
import cv2
import os

# Load video frpm PATH if CAMERA is OFF
if CAMERA_FLAG == False:
    VIDEOPATH = os.path.join(os.getcwd(), FOLDERNAME, VIDEONAME)
else:
    VIDEOPATH = 0

class SODV:
    def __init__(self, source = VIDEOPATH, distance = DISTANCE, START = True):

        self.video = cv2.VideoCapture(source)
        self.model = Model(utilsdir=UTILSDIR, modeldir=MODELDIR, weights=WEIGHTS, cfg=CFG, coco=COCONAMES)    
        self.distance = distance        
        if START == True: self.main()
    

    # @param *args: (xmin, ymin, xmax, ymax)
    def calculate_centroid(self, *args):
        # Center point of bounding boxes
        return (((args[2] + args[0])/2), ((args[3] + args[1])/2))
    
    # @param *args: (xcenter_1, ycenter_1, xcenter_2, ycenter_2)
    def calculate_euclidean_distance(self, *args):
        # Euclidean distance
        return math.sqrt((args[0]-args[2])**2 + (args[1]-args[3])**2)
    
    # @param *args: (frame, xmin, ymin, xmax, ymax, color)
    def rect_detection_box(self, *args):
        cv2.rectangle(args[0], (args[1], args[2]), (args[3], args[4]), args[5], 1)
    
    def main(self):
        try:
            net, layerNames, classes = self.model.predict()
            print('[PASSED] Model loaded.')
        except Exception:
            print('[FAILED] Unable to load model.')
        
        while (self.video.isOpened()):
            
            high_counter, low_counter = 0, 0
            centroids = list()
            detected_bbox_colors = list()
            detected_bbox = list()
            self.flag, self.frame = self.video.read() 

            # Resize frame for prediction 
            if self.flag:
                self.frameResized = cv2.resize(self.frame, (416, 416))       
            else:
                break

            height, width, channels = self.frame.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(self.frameResized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(layerNames)

            confidences = list()
            boxes = list()

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # If class is not 0 which is person, ignore it.
                    if classID != 0:
                        continue

                    # If prediction is more than 50% 
                    if confidence > CONFIDENCE:
                        
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)

                        # Bbox width and height
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Bbox x and y axis pixel coordinate
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))

            # Apply non-max suppression (NMS)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]

                    xmn = x
                    ymn = y
                    xmx = (x + w)
                    ymx = (y + h)

                    # Calculate centroid point for bbox (detected_bbox)
                    centroid = self.calculate_centroid(xmn, ymn, xmx, ymx)
                    detected_bbox.append([xmn, ymn, xmx, ymx, centroid])

                    violation = False
                    for k in range (len(centroids)):
                        c = centroids[k]
                        
                        # Compare pixel distance between bbox (detected_bbox)
                        if self.calculate_euclidean_distance(c[0], c[1], centroid[0], centroid[1]) <= self.distance:
                            detected_bbox_colors[k] = True
                            violation = True
                            cv2.line(self.frame, (int(c[0]), int(c[1])), (int(centroid[0]), int(centroid[1])), YELLOW, 1, cv2.LINE_AA)
                            cv2.circle(self.frame, (int(c[0]), int(c[1])), 3, ORANGE, -1,cv2.LINE_AA)
                            cv2.circle(self.frame, (int(centroid[0]), int(centroid[1])), 3, ORANGE, -1, cv2.LINE_AA)
                            break
                    centroids.append(centroid)
                    detected_bbox_colors.append(violation)        

            for i in range (len(detected_bbox)):
                xmin = detected_bbox[i][0]
                ymin = detected_bbox[i][1]
                xmax = detected_bbox[i][2]
                ymax = detected_bbox[i][3]
                
                if detected_bbox_colors[i] == False:
                    self.rect_detection_box(self.frame, xmin, ymin, xmax, ymax, BLACK)
                    label = "LOW"
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    ylabel = max(ymin, labelSize[1])
                    cv2.rectangle(self.frame, (xmin, ylabel - labelSize[1]), (xmin + labelSize[0], ymin + baseLine), BLACK, cv2.FILLED)
                    cv2.putText(self.frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
                    low_counter += 1

                else:
                    self.rect_detection_box(self.frame, xmin, ymin, xmax, ymax, RED)
                    label = "HIGH"
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    ylabel = max(ymin, labelSize[1])
                    cv2.rectangle(self.frame, (xmin, ylabel - labelSize[1]),(xmin + labelSize[0], ymin + baseLine), RED, cv2.FILLED)
                    cv2.putText(self.frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1, cv2.LINE_AA)
                    high_counter += 1

            cv2.rectangle(self.frame, (13, 10),(250, 60), GREY, cv2.FILLED)
            LINE = "--"
            HIGHRISK_TEXT = f'HIGH RISK: {high_counter} people'
            cv2.putText(self.frame, LINE, (28, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2, cv2.LINE_AA)
            cv2.putText(self.frame, HIGHRISK_TEXT, (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1, cv2.LINE_AA)

            LOWRISK_TEXT = f'LOW RISK: {low_counter} people'
            cv2.putText(self.frame, LINE, (28, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2, cv2.LINE_AA)
            cv2.putText(self.frame, LOWRISK_TEXT, (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1, cv2.LINE_AA)

            cv2.imshow("SODV", self.frame)

            if cv2.waitKey(1) >= 0:  
                break

        self.video.release()

if __name__ == '__main__':
    start_time = time()
    SODV()
    print(f'Finished after {round(time()-start_time, 2)}s')
    cv2.destroyAllWindows()