from config.model import Model
from config.config import *
import numpy as np
import threading
import time
import math
import cv2
import os
import matplotlib.pyplot as plt

class App:
    '''
    Social Distancing Violation System SODV
    =======================================
    '''
    def __init__(self, source, distance, input_information, start = True):
        self.input_information = input_information
        self.video = cv2.VideoCapture(source)
        self.model = Model(utilsdir=UTILSDIR, modeldir=MODELDIR, weights=WEIGHTS, cfg=CFG, labelsdir=LABELSDIR, coco=COCONAMES)    
        self.distance = distance
        if start: self.main()        

    def calculate_centroid(self, *args):
        '''
        Getting the center point of ground plane for the bounding box (bbox)
        --------------------------------------------------------------------
        - param : args : (xmin, ymin, xmax, ymax)
        - xmin : x-axis minimum value
        - ymin : y-axis minimum value
        - xmax : x-axis maximum value
        - ymax : y-axis maximum value

        - return value : C(x,y), the center of bounding box ground plane 

        - To return bbox center point: 
        - return (((args[2] + args[0])/2), ((args[3] + args[1])/2))
        '''
        return (((args[2] + args[0])/2), args[3])
    
    def calculate_euclidean_distance(self, *args):        
        '''
        Euclidean Distance
        ------------------
        - param : args : (xmin, ymin, xmax, ymax)
        - xcenter_1, ycenter_1 : C1(x,y)
        - xcenter_2, ycenter_2 : C2(x,y)

        return value : D(C1,C2)
        '''
        return math.sqrt((args[0]-args[2])**2 + (args[1]-args[3])**2)
    
    def rect_detection_box(self, *args):
        '''
        Bounding Box (bbox)
        -------------------
        - param : args : (frame, xmin, ymin, xmax, ymax, color)
        - frame : continuos frame stream
        - xmin  : x-axis minimum value
        - ymin  : y-axis minimum value
        - xmax  : x-axis maximum value
        - ymax  : y-axis maximum value
        - color : color for the bounding box
        '''
        cv2.rectangle(args[0], (args[1], args[2]), (args[3], args[4]), args[5], 1)

    def information_display(self):
        '''
        Display violation detection information
        ---------------------------------------
        '''
        cv2.rectangle(self.frame, (13, 5), (250, 30), BLACK, cv2.FILLED)
        cv2.putText(self.frame, f'{self.input_information}', (28, 24), cv2.FONT_HERSHEY_DUPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

        cv2.rectangle(self.frame, (13, 30), (250, 80), GREY, cv2.FILLED)
        LINE = "--"
        HIGHRISK_TEXT = f'HIGH RISK: {self.high_counter} people'
        cv2.putText(self.frame, LINE, (28, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, RED, 2, cv2.LINE_AA)
        cv2.putText(self.frame, HIGHRISK_TEXT, (60, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, BLUE, 1, cv2.LINE_AA)

        LOWRISK_TEXT = f'LOW RISK: {self.low_counter} people'
        cv2.putText(self.frame, LINE, (28, 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, BLACK, 2, cv2.LINE_AA)
        cv2.putText(self.frame, LOWRISK_TEXT, (60, 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, BLUE, 1, cv2.LINE_AA)
    
    def generate_chart(self):
        '''
        Dashboard
        ---------
        '''
        fig, ax = plt.subplots(figsize=(4,4))
        ax.pie([self.high_counter, self.low_counter], labels = [f'High risk: {self.high_counter}', f'Low risk: {self.low_counter}'], colors=[RED_DB, GREEN_DB])
        ax.legend()
        plt.savefig(DASHBOARD, transparent=True, dpi=700)
        plt.close()

    def main(self):
        net, layerNames, classes = self.model.predict()
        while (self.video.isOpened()):
            self.high_counter, self.low_counter = 0, 0
            centroids = list()
            detected_bbox_colors = list()
            detected_bbox = list()
 
            self.flag, self.frame = self.video.read()

            if THREAD:
                try:
                    self.thread_1 = threading.Thread(target=self.video.read)
                    self.thread_1.daemon = True
                    self.thread_1.start()
                except RuntimeError as err:
                    print(err)
            else:
                pass
            active_thread_count = int(threading.activeCount())  

            # Resize frame for prediction 
            if self.flag:
                self.frameResized = cv2.resize(self.frame, (416, 416))       
            else:
                break

            # Detecting objects
            blob = cv2.dnn.blobFromImage(self.frameResized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(layerNames)

            confidences = list()
            boxes = list()
            height, width, _ = self.frame.shape
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
                        # Bbox x and y axis coordinate
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
                    # Calculate ground plane center point of bbox (detected_bbox)
                    centroid = self.calculate_centroid(xmn, ymn, xmx, ymx)
                    detected_bbox.append([xmn, ymn, xmx, ymx, centroid])

                    violation = False
                    for k in range (len(centroids)):
                        # Compare the distance of center point with each other
                        if self.calculate_euclidean_distance(centroids[k][0], centroids[k][1], centroid[0], centroid[1]) <= self.distance:
                            detected_bbox_colors[k] = True
                            violation = True
                            cv2.line(self.frame, (int(centroids[k][0]), int(centroids[k][1])), (int(centroid[0]), int(centroid[1])), YELLOW, 1, cv2.LINE_AA)
                            cv2.circle(self.frame, (int(centroids[k][0]), int(centroids[k][1])), 3, ORANGE, -1,cv2.LINE_AA)
                            cv2.circle(self.frame, (int(centroid[0]), int(centroid[1])), 3, ORANGE, -1, cv2.LINE_AA)
                            break
                    centroids.append(centroid)
                    detected_bbox_colors.append(violation)        

            for i in range (len(detected_bbox)):
                xmin = detected_bbox[i][0]
                ymin = detected_bbox[i][1]
                xmax = detected_bbox[i][2]
                ymax = detected_bbox[i][3]
                
                # Else, wrap red bbox
                if detected_bbox_colors[i]:
                    self.rect_detection_box(self.frame, xmin, ymin, xmax, ymax, RED)
                    label = "high".upper()
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)

                    ylabel = max(ymin, labelSize[1])
                    cv2.rectangle(self.frame, (xmin, ylabel - labelSize[1]),(xmin + labelSize[0], ymin + baseLine), RED, cv2.FILLED)
                    cv2.putText(self.frame, label, (xmin, ymin), cv2.FONT_HERSHEY_DUPLEX, 0.5, ORANGE, 1, cv2.LINE_AA)
                    self.high_counter += 1

                # If euclidean distance less than (<) DISTANCE, wrap black bbox
                else:
                    self.rect_detection_box(self.frame, xmin, ymin, xmax, ymax, BLACK)
                    label = "low".upper()
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)

                    ylabel = max(ymin, labelSize[1])
                    cv2.rectangle(self.frame, (xmin, ylabel - labelSize[1]), (xmin + labelSize[0], ymin + baseLine), BLACK, cv2.FILLED)
                    cv2.putText(self.frame, label, (xmin, ymin), cv2.FONT_HERSHEY_DUPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
                    self.low_counter += 1
                    
            if DASHBOARD_FLAG:
                self.generate_chart()
                self.dashboard = cv2.imread(DASHBOARD)
                cv2.namedWindow("Dashboard: SODV", cv2.WINDOW_NORMAL)
                cv2.imshow("Dashboard: SODV", self.dashboard)
            else:
                pass
            self.information_display()

            # Resizable windows
            cv2.namedWindow("SODV: Social Distancing Violation System", cv2.WINDOW_NORMAL)
            cv2.imshow("SODV: Social Distancing Violation System", self.frame)
             
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print(f'[THREAD] Active thread used: {active_thread_count}')      
        self.video.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    start_time = time.time()
    if CAMERA_FLAG:
        VIDEOPATH = CAMERA_ID
        VIDEO_IND = f'camera_id_{CAMERA_ID}'.upper()
    else:
        VIDEOPATH = os.path.join(os.getcwd(), FOLDERNAME, VIDEONAME)
        VIDEO_IND = VIDEONAME[:-4].upper()
    App(VIDEOPATH, DISTANCE, VIDEO_IND)
    print(f'[STATUS] Finished after {round(time.time()-start_time, 2)}s')