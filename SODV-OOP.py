__author__ = "Afiq Harith"
__email__ = "afiqharith05@gmail.com"
__date__ = "08 Oct 2020"
__status__ = "Development-OOP"

import cv2
import numpy as np
import math
import os
from setup.LoadModel import LoadModel
from setup.config import Config as cfg

GREEN = cfg.colors('GREEN')
RED = cfg.colors('RED')
YELLOW = cfg.colors('YELLOW')
WHITE = cfg.colors('WHITE')
ORANGE = cfg.colors('ORANGE')
BLUE = cfg.colors('BLUE')
GREY = cfg.colors('GREY')

class SODV:
    def __init__(self,videoSRC,distance):
        self.video = cv2.VideoCapture(videoSRC)
        self.distance = distance
        self.main()
        
    def calculateCentroid(self, xmin,ymin,xmax,ymax):
        xmid = ((xmax+xmin)/2)
        ymid = ((ymax+ymin)/2)
        centroid = (xmid,ymid)

        return xmid,ymid,centroid
    
    def get_distance(self,x1,x2,y1,y2):
        distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        
        return distance
    
    def draw_detection_box(self,frame,x1,y1,x2,y2,color):
        cv2.rectangle(frame,(x1,y1),(x2,y2), color, 2)
    
    def main(self):
        net, output_layers, classes = LoadModel.get()

        while True:
            stat_H, stat_L = 0, 0
            centroids = []
            box_colors = []
            detectedBox = []
            ret, frame = self.video.read() 

            if ret:
                frame_resized = cv2.resize(frame, (416,416)) # resize frame for prediction       
            else:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
            height, width, channels = frame.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    #if prediction is 50% and class id is 0 which is 'person'
                    if confidence > 0.5 and class_id == 0:
                        
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
                        class_ids.append(class_id)

            # apply non-max suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]

                    xmin = x
                    ymin = y
                    xmax = (x + w)
                    ymax = (y + h)

                    #calculate centroid point for bounding boxes
                    xmid, ymid, centroid = self.calculateCentroid(xmin,ymin,xmax,ymax)
                    detectedBox.append([xmin,ymin,xmax,ymax,centroid])

                    my_color = 0
                    for k in range (len(centroids)):
                        c = centroids[k]
                        
                        if self.get_distance(c[0],centroid[0],c[1],centroid[1]) <= self.distance:
                            box_colors[k] = 1
                            my_color = 1
                            cv2.line(frame, (int(c[0]),int(c[1])), (int(centroid[0]),int(centroid[1])), YELLOW, 1,cv2.LINE_AA)
                            cv2.circle(frame, (int(c[0]),int(c[1])), 3, ORANGE, -1,cv2.LINE_AA)
                            cv2.circle(frame, (int(centroid[0]),int(centroid[1])), 3, ORANGE, -1,cv2.LINE_AA)
                            break
                    centroids.append(centroid)
                    box_colors.append(my_color)        

            for i in range (len(detectedBox)):
                x1 = detectedBox[i][0]
                y1 = detectedBox[i][1]
                x2 = detectedBox[i][2]
                y2 = detectedBox[i][3]
                
                #for ellipse output
                xc = ((x2+x1)/2)
                yc = y2-5
                centroide = (int(xc),int(yc))

                if box_colors[i] == 0:
                    color = WHITE
                    self.draw_detection_box(frame,x1,y1,x2,y2,color)
                    label = "safe"
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                    y1label = max(y1, labelSize[1])
                    cv2.rectangle(frame, (x1, y1label - labelSize[1]),(x1 + labelSize[0], y1 + baseLine), WHITE, cv2.FILLED)
                    cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1,cv2.LINE_AA)
                    stat_L += 1

                else:
                    color = RED
                    self.draw_detection_box(frame,x1,y1,x2,y2,color)
                    label = "unsafe"
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                    y1label = max(y1, labelSize[1])
                    cv2.rectangle(frame, (x1, y1label - labelSize[1]),(x1 + labelSize[0], y1 + baseLine), WHITE, cv2.FILLED)
                    cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1,cv2.LINE_AA)
                    stat_H += 1

            cv2.rectangle(frame, (13, 10),(250, 60), GREY, cv2.FILLED)
            LINE = "--"
            INDICATION_H = f'HIGH RISK: {str(stat_H)} people'
            cv2.putText(frame, LINE, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1,cv2.LINE_AA)
            cv2.putText(frame, INDICATION_H, (60,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1,cv2.LINE_AA)

            INDICATION_L = f'LOW RISK : {str(stat_L)} people'
            cv2.putText(frame, LINE, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1,cv2.LINE_AA)
            cv2.putText(frame, INDICATION_L, (60,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1,cv2.LINE_AA)

            cv2.imshow("SODV", frame)

            if cv2.waitKey(1) >= 0:  
                break

        self.video.release()

if __name__ == '__main__':

    # Load video
    videoName = "TownCentre.mp4"
    videosPath = "videos/"
    videoSRC = os.path.join(os.getcwd(), videosPath, videoName)
    distance, *_ = cfg.get2Data(videoName)

    SODV(videoSRC,distance)
    cv2.destroyAllWindows()