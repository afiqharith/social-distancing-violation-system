from utils.refresh_logging_data import Initilization
import scipy.spatial.distance as dst
from config.config import Config
import matplotlib.pyplot as plt
from config.model import Model
import numpy as np
import threading
import tabulate
import datetime
import json
import time
import math
import cv2
import os
c = Config()

class Pipeline:
    '''
    Pipeline SODV
    =============
    '''
    temp_log = os.path.join(os.getcwd(), 'temp', 'logging.json')
    start_time = time.perf_counter()
    if c.CAMERA_FLAG:
        VIDEOPATH = c.CAMERA_ID
        VIDEO_IND = f'camera_id_{c.CAMERA_ID}'.upper()
    else:
        VIDEOPATH = os.path.join(os.getcwd(), c.FOLDERNAME, c.VIDEONAME)
        VIDEO_IND = c.VIDEONAME[:-4].upper()

    def __init__(self, source=VIDEOPATH, distance=c.DISTANCE, input_information=VIDEO_IND, start_time=start_time, temp_log_path=temp_log):
        Initilization()
        self.start_time = start_time
        self.temp_log =  temp_log_path
        self.input_information = input_information
        self.video = cv2.VideoCapture(source)
        self.model = Model(utilsdir=c.UTILSDIR, modeldir=c.MODELDIR, weights=c.WEIGHTS, cfg=c.CFG, labelsdir=c.LABELSDIR, coco=c.COCONAMES)
        self.distance = distance
        self.active_thread_count = 0
        self.pTime = 0
        self.frame_counter = 0

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
    
    def cross_line(self, xmin, ymin, xmax, ymax, color):
        '''
        Use cross line as detection output instead of boxes: dev_process(trial)
        -----------------------------------------------------------------------
        - horizontal = s(xmin, yc), e(xmax, yc)
        - verticle = s(xc, ymin), e(xc, ymax)
        '''
        xc = (xmin+xmax)/2
        yc = (ymin+ymax)/2

        cv2.line(self.frame, (int(xmin), int(yc)), (int(xmax), int(yc)), color, 1, cv2.LINE_AA)
        cv2.line(self.frame, (int(xc), int(ymin)), (int(xc), int(ymax)), color, 1, cv2.LINE_AA)

    def information_display(self):
        '''
        Display violation detection information
        ---------------------------------------
        '''
        cv2.rectangle(self.frame, (13, 5), (250, 30), c.BLACK, cv2.FILLED)
        cv2.putText(self.frame, f'{self.input_information}', (28, 24), cv2.FONT_HERSHEY_DUPLEX, 0.5, c.WHITE, 1, cv2.LINE_AA)
        cv2.putText(self.frame, f'{self.fps}fps', (200, 24), cv2.FONT_HERSHEY_DUPLEX, 0.5, c.GREEN, 1, cv2.LINE_AA)

        cv2.rectangle(self.frame, (13, 30), (250, 80), c.GREY, cv2.FILLED)
        LINE = "--"
        HIGHRISK_TEXT = f'HIGH RISK: {self.high_counter} people'
        cv2.putText(self.frame, LINE, (28, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, c.RED, 2, cv2.LINE_AA)
        cv2.putText(self.frame, HIGHRISK_TEXT, (60, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, c.BLUE, 1, cv2.LINE_AA)

        LOWRISK_TEXT = f'LOW RISK: {self.low_counter} people'
        cv2.putText(self.frame, LINE, (28, 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, c.BLACK, 2, cv2.LINE_AA)
        cv2.putText(self.frame, LOWRISK_TEXT, (60, 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, c.BLUE, 1, cv2.LINE_AA)
    
    def generate_fps(self):
        '''
        Frame per second (fps) counter
        ------------------------------
        '''
        self.cTime = time.time()
        fps = 1/(self.cTime - self.pTime)
        self.pTime = self.cTime
        return int(fps)

    def generate_chart(self):
        '''
        Dashboard
        ---------
        '''
        fig, ax = plt.subplots(figsize=(4,4))
        ax.pie([self.high_counter, self.low_counter], labels = [f'High risk: {self.high_counter}', f'Low risk: {self.low_counter}'], colors=[c.RED_DB, c.GREEN_DB])
        ax.legend()
        plt.savefig(c.DASHBOARD, transparent=True, dpi=700)
        plt.close()

    def generate_logging(self):
        '''
        Generate logging for the video
        ------------------------------
        '''
        with open(self.temp_log) as fileIn:
            loaded = json.load(fileIn)
            data = loaded['data']
        if self.frame_counter == 1 or self.frame_counter%5 == 0:
            items = {
                "time" : f'{self.log_time}',
                "frames" : int(self.frame_counter),
                "high_risk": int(self.high_counter),
                "low_risk": int(self.low_counter)
            }
            data.append(items)

        try:
            with open(self.temp_log, 'w') as fileOut:
                json.dump(loaded, fileOut, sort_keys=True)
        except IOError as e:
            print(e)
    
    def show_logging(self):
        '''
        Display logging for the video after video is finished
        -----------------------------------------------------
        '''
        with open(self.temp_log, 'r') as fileIn:
            loaded = json.load(fileIn)
            data = loaded['data']
        time, frame, high, low = list(), list(), list(), list()
        to_display = {
            "Time" : time,
            "Frame": frame,
            "High": high,
            "Low": low
        }
        for i in data:
            time.append(i['time'])
            frame.append(i['frames'])
            high.append(i['high_risk'])
            low.append(i['low_risk'])
        return tabulate.tabulate(to_display, headers="keys", tablefmt="pretty")
    
    def show_usage(self):
        '''
        Display thread usage for the video after video is finished
        ----------------------------------------------------------
        '''
        elapsed = time.perf_counter()-self.start_time
        data = {"Active thread used": [self.active_thread_count],
                "Status": [f'Executed in {elapsed:.2f}s']}
        return tabulate.tabulate(data, headers="keys", tablefmt="pretty")

    def __str__(self):
        return f'Output Data =>\n{self.show_logging()}\nHardware usage =>\n{self.show_usage()}'

class App(Pipeline):
    '''
    Social Distancing Violation System SODV
    =======================================
    '''
    def __init__(self):
        super().__init__()
        self.net, self.layerNames, _ = self.model.predict()
        self.windowName = "SODV: Social Distancing Violation System"
        self.main()

    def main(self):
        while (self.video.isOpened()):
            self.high_counter, self.low_counter = 0, 0
            centroids = list()
            detected_bbox_colors = list()
            detected_bbox = list()
 
            self.flag, self.frame = self.video.read()

            if c.THREAD:
                try:
                    self.thread_1 = threading.Thread(target=self.video.read)
                    self.thread_1.daemon = True
                    self.thread_1.start()
                except RuntimeError as err:
                    print(err)
            else:
                pass
            self.active_thread_count = int(threading.activeCount())

            # Resize frame for prediction 
            if self.flag:
                self.frameResized = cv2.resize(self.frame, (416, 416))
            else:
                break

            # Detecting objects
            blob = cv2.dnn.blobFromImage(self.frameResized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            layerOutputs = self.net.forward(self.layerNames)

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
                    if confidence > c.CONFIDENCE:
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
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, c.CONFIDENCE, c.THRESHOLD)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    xmn = x
                    ymn = y
                    xmx = (x + w)
                    ymx = (y + h)
                    # Calculate ground plane center point of bbox (detected_bbox)
                    centroid = self.calculate_centroid(xmn, ymn, xmx, ymx)
                    detected_bbox.append([xmn, ymn, xmx, ymx])

                    isViolation = False
                    for k in range (len(centroids)):
                        # Compare the distance of center point with each other
                        if dst.euclidean([centroids[k][0], centroids[k][1]], [centroid[0], centroid[1]]) <= self.distance:
                            detected_bbox_colors[k] = True
                            isViolation = True
                            cv2.line(self.frame, (int(centroids[k][0]), int(centroids[k][1])), (int(centroid[0]), int(centroid[1])), c.YELLOW, 1, cv2.LINE_AA)
                            cv2.circle(self.frame, (int(centroids[k][0]), int(centroids[k][1])), 3, c.ORANGE, -1,cv2.LINE_AA)
                            cv2.circle(self.frame, (int(centroid[0]), int(centroid[1])), 3, c.ORANGE, -1, cv2.LINE_AA)
                            break
                    centroids.append(centroid)
                    detected_bbox_colors.append(isViolation)

            for i in range (len(detected_bbox)):
                xmin = detected_bbox[i][0]
                ymin = detected_bbox[i][1]
                xmax = detected_bbox[i][2]
                ymax = detected_bbox[i][3]
                self.cross_line(xmin, ymin, xmax, ymax, c.GREY)
                # Else, wrap red bbox
                if detected_bbox_colors[i]:
                    # self.cross_line(xmin, ymin, xmax, ymax, c.RED)
                    self.rect_detection_box(self.frame, xmin, ymin, xmax, ymax, c.RED)
                    label = "high".upper()
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)

                    ylabel = max(ymin, labelSize[1])
                    cv2.rectangle(self.frame, (xmin, ylabel - labelSize[1]),(xmin + labelSize[0], ymin + baseLine), c.RED, cv2.FILLED)
                    cv2.putText(self.frame, label, (xmin, ymin), cv2.FONT_HERSHEY_DUPLEX, 0.5, c.ORANGE, 1, cv2.LINE_AA)
                    self.high_counter += 1

                # If euclidean distance less than (<) DISTANCE, wrap black bbox
                else:
                    # self.cross_line(xmin, ymin, xmax, ymax, c.BLACK)
                    self.rect_detection_box(self.frame, xmin, ymin, xmax, ymax, c.BLACK)
                    label = "low".upper()
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)

                    ylabel = max(ymin, labelSize[1])
                    cv2.rectangle(self.frame, (xmin, ylabel - labelSize[1]), (xmin + labelSize[0], ymin + baseLine), c.BLACK, cv2.FILLED)
                    cv2.putText(self.frame, label, (xmin, ymin), cv2.FONT_HERSHEY_DUPLEX, 0.5, c.GREEN, 1, cv2.LINE_AA)
                    self.low_counter += 1
                    
            if c.DASHBOARD_FLAG:
                self.generate_chart()
                self.dashboard = cv2.imread(c.DASHBOARD)
                cv2.namedWindow(f'SODV Dashboard: {self.input_information}', cv2.WINDOW_NORMAL)
                cv2.imshow(f'SODV Dashboard: {self.input_information}', self.dashboard)
            else:
                pass
            self.fps = self.generate_fps()
            self.information_display()
            self.frame_counter += 1
            self.log_time = datetime.datetime.now().strftime("%d-%m-%Y %I:%M:%S%p")
            self.generate_logging()
            # Resizable windows
            cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
            cv2.imshow(self.windowName, self.frame)
             
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE) <1:
                break

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = App()
    if c.LOGGING:
        print(app)