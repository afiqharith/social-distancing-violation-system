from utils.refresh_logging_data import Initilization
from config.config import Config as c
import scipy.spatial.distance as dst
import matplotlib.pyplot as plt
from config.model import Model
import numpy as np
import threading
import tabulate
import datetime
import json
import time
import cv2
import os

class LoadInfoFromDisk:
    '''
    Loading all informations from disk
    ==================================
    '''
    def __init__(self) -> None:
        self.temp_log: str = os.path.join(os.getcwd(), 'temp', 'logging.json')
        self.start_time: float = time.perf_counter()
        self.distance: float = c.DISTANCE

        if c.CAMERA_FLAG:
            self.camera_on()
        else:
            self.camera_off()

    def camera_on(self) -> None:
        '''
        Get the camera ID for the camera stream
        '''
        self.source = c.CAMERA_ID
        self.input_information = f'camera_id_{c.CAMERA_ID}'.upper()
    
    def camera_off(self) -> None:
        '''
        Get the path for the video stream
        '''
        self.source = os.path.join(os.getcwd(), c.FOLDERNAME, c.VIDEONAME)
        self.input_information = c.VIDEONAME[:-4].upper()

class ProgramFeatures(LoadInfoFromDisk):
    '''
    Features for the SODV
    =====================
    '''
    def __init__(self) -> None:
        Initilization()
        super().__init__()
        self.video = cv2.VideoCapture(self.source)
        self.model = Model(utilsdir=c.UTILSDIR, modeldir=c.MODELDIR, weights=c.WEIGHTS, cfg=c.CFG, labelsdir=c.LABELSDIR, coco=c.COCONAMES, cuda=False)
        self.active_thread_count: int = None
        self.p_time: float = 0
        self.frame_counter: int = 0

    def calculate_centroid(self, *axis: int) -> tuple:
        '''
        Getting the center point of ground plane for the bounding box (bbox)
        --------------------------------------------------------------------
        - param : *axis : (xmin_pre_process, ymin_pre_process, xmax_pre_process, ymax_pre_process)
        - xmin_pre_process : x-axis minimum value
        - ymin_pre_process : y-axis minimum value
        - xmax_pre_process : x-axis maximum value
        - ymax_pre_process : y-axis maximum value

        - return value : C(x,y), the center of bounding box ground plane 

        - To return bbox center point: 
        - return (((axis[2] + axis[0])/2), ((axis[3] + axis[1])/2))
        '''
        return (((axis[2] + axis[0])/2), axis[3])
    
    def rect_detection_box(self, color: tuple, *axis: int) -> None:
        '''
        Bounding Box (bbox)
        -------------------
        - param : color, *args : (color, xmin, ymin, xmax, ymax)
        - color : color for the bounding box
        - xmin  : x-axis minimum value
        - ymin  : y-axis minimum value
        - xmax  : x-axis maximum value
        - ymax  : y-axis maximum value
        '''
        cv2.rectangle(self.frame, (axis[0], axis[1]), (axis[2], axis[3]), color, 1)
    
    def cross_line(self, xmin: int, ymin: int, xmax: int, ymax: int, color: tuple) -> None:
        '''
        Use cross line as detection output instead of boxes: dev_process(trial)
        -----------------------------------------------------------------------
        - horizontal = s(xmin, yc), e(xmax, yc)
        - verticle = s(xc, ymin), e(xc, ymax)
        '''
        xcenter = (xmin + xmax)/2
        ycenter = (ymin + ymax)/2

        cv2.line(self.frame, (int(xmin), int(ycenter)), (int(xmax), int(ycenter)), color, 1, cv2.LINE_AA)
        cv2.line(self.frame, (int(xcenter), int(ymin)), (int(xcenter), int(ymax)), color, 1, cv2.LINE_AA)
    
    def draw_line_distance_between_bbox(self, centroid_new: tuple, centroid_old: tuple) -> None:
        '''
        Display line between violated bbox
        ----------------------------------
        '''
        cv2.line(self.frame, (int(centroid_new[0]), int(centroid_new[1])), (int(centroid_old[0]), int(centroid_old[1])), c.YELLOW, 1, cv2.LINE_AA)
        cv2.circle(self.frame, (int(centroid_new[0]), int(centroid_new[1])), 3, c.ORANGE, -1, cv2.LINE_AA)
        cv2.circle(self.frame, (int(centroid_old[0]), int(centroid_old[1])), 3, c.ORANGE, -1, cv2.LINE_AA)

    def status_display(self, label: str, xmin: int, ymin: int, **color: tuple) -> None:
        '''
        Display violation status on top of bbox
        ---------------------------------------
        '''
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        ylabel = max(ymin, label_size[1])
        cv2.rectangle(self.frame, (xmin, ylabel - label_size[1]), (xmin + label_size[0], ymin + base_line), color['background'], cv2.FILLED)
        cv2.putText(self.frame, label, (xmin, ymin), cv2.FONT_HERSHEY_DUPLEX, 0.5, color['font'], 1, cv2.LINE_AA)

    def information_display(self) -> None:
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
    
    def dashboard_display(self) -> None:
        '''
        Display dashboard
        -----------------
        '''
        self.dashboard = cv2.imread(c.DASHBOARD_PATH)
        cv2.namedWindow(f'SODV Dashboard: {self.input_information}', cv2.WINDOW_NORMAL)
        cv2.imshow(f'SODV Dashboard: {self.input_information}', self.dashboard)

    def generate_fps(self) -> int:
        '''
        Frame per second (fps) counter
        ------------------------------
        '''
        self.c_time = time.time()
        fps = 1 / (self.c_time - self.p_time)
        self.p_time = self.c_time
        return int(fps)

    def generate_chart(self) -> None:
        '''
        Generating chart and save the chart image
        -----------------------------------------
        '''
        fig, ax = plt.subplots(figsize=(4,4))
        ax.pie([self.high_counter, self.low_counter], labels = [f'High risk: {self.high_counter}', f'Low risk: {self.low_counter}'], colors=[c.RED_DB, c.GREEN_DB])
        ax.legend()
        plt.savefig(c.DASHBOARD_PATH, transparent=True, dpi=700)
        plt.close()

    def generate_logging(self) -> None:
        '''
        Generate logging for the video
        ------------------------------
        '''
        with open(self.temp_log) as file_in:
            loaded = json.load(file_in)
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
            with open(self.temp_log, 'w') as file_out:
                json.dump(loaded, file_out, sort_keys=True)
        except IOError as e:
            print(e)
    
    def show_logging(self) -> str:
        '''
        Display logging for the video after video is finished
        -----------------------------------------------------
        '''
        with open(self.temp_log, 'r') as file_in:
            loaded = json.load(file_in)
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
    
    def show_usage(self) -> str:
        '''
        Display thread usage for the video after video is finished
        ----------------------------------------------------------
        '''
        elapsed = time.perf_counter()-self.start_time
        data = {"Active thread used": [self.active_thread_count],
                "Status": [f'Executed in {elapsed:.2f}s']}
        return tabulate.tabulate(data, headers="keys", tablefmt="pretty")

    def __str__(self) -> str:
        return f'Output Data =>\n{self.show_logging()}\nHardware usage =>\n{self.show_usage()}'

class App(ProgramFeatures):
    '''
    Social Distancing Violation System SODV
    =======================================
    A camera tests program to identifie persons who are not adhering to COVID social distancing measures
    '''
    def __init__(self) -> None:
        super().__init__()
        self.net, self.layer_names, _ = self.model.network, self.model.layer_names, self.model.classes
        self.window_name = "SODV: Social Distancing Violation System"
        self.main()

    def main(self) -> None:
        while (self.video.isOpened()):
            self.high_counter, self.low_counter = 0, 0
 
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

            if self.flag:
                '''
                Resize the frame for the object prediction
                '''
                self.frame_resized = cv2.resize(self.frame, (416, 416))
            else:
                break

            '''
            Detecting objects in the resized frame
            '''
            blob = cv2.dnn.blobFromImage(self.frame_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            layer_outputs = self.net.forward(self.layer_names)

            confidences = list()
            boxes = list()
            frame_height, frame_width, _ = self.frame.shape
            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_ID = np.argmax(scores)
                    confidence = scores[class_ID]

                    '''
                    Ignore the object classes except 'person'
                    '''
                    if class_ID != 0:
                        continue

                    '''
                    Get the bbox axis point if the conficence threshold of detected object class is more than 50%
                    '''
                    if confidence > c.MIN_CONFIDENCE:
                        '''
                        Get the bbox center point of detected object class
                        '''
                        center_x = int(detection[0] * frame_width)
                        center_y = int(detection[1] * frame_height)

                        '''
                        Get the bbox width and height of detected object class
                        '''
                        bbox_cluster_w = int(detection[2] * frame_width)
                        bbox_cluster_h = int(detection[3] * frame_height)

                        '''
                        Compute the bbox minimum x and y axis point of detected object class
                        '''
                        bbox_cluster_x = int(center_x - bbox_cluster_w / 2)
                        bbox_cluster_y = int(center_y - bbox_cluster_h / 2)

                        boxes.append([bbox_cluster_x, bbox_cluster_y, bbox_cluster_w, bbox_cluster_h])
                        confidences.append(float(confidence))

            '''
            Apply non-max suppression (NMS) for the clustered detected object class bbox
            '''
            centroids = list()
            detected_bboxes = list()
            detected_bbox_colors = list()
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, c.MIN_CONFIDENCE, c.NMS_THRESHOLD)
            for i, box in enumerate(boxes):
                if i in indexes:
                    x, y, w, h = box
                    xmin_pre_process = x
                    ymin_pre_process = y
                    xmax_pre_process = (x + w)
                    ymax_pre_process = (y + h)

                    '''
                    Compute the ground plane center point for the bbox
                    '''
                    centroid_new = self.calculate_centroid(xmin_pre_process, ymin_pre_process, xmax_pre_process, ymax_pre_process)
                    detected_bboxes.append([xmin_pre_process, ymin_pre_process, xmax_pre_process, ymax_pre_process])

                    is_violation = None
                    for j, centroid_old in enumerate(centroids):
                        '''
                        Compute the pair-wise distance for each bbox center point using euclidean distance
                        '''
                        if dst.euclidean([centroid_new[0], centroid_new[1]], [centroid_old[0], centroid_old[1]]) <= self.distance:
                            detected_bbox_colors[j] = True
                            is_violation = True
                            self.draw_line_distance_between_bbox(centroid_new, centroid_old)
                            break
                    centroids.append(centroid_new)
                    detected_bbox_colors.append(is_violation)

            for i, detected_bbox in enumerate(detected_bboxes):
                xmin = detected_bbox[0]
                ymin = detected_bbox[1]
                xmax = detected_bbox[2]
                ymax = detected_bbox[3]

                '''
                Compute the euclidean distance between the pairwise bbox and display the bbox output
                - Wrap with black bbox if the pairwise not violate social distance measure
                - Wrap with red bbox if the pairwise violate social distance measure
                '''
                if detected_bbox_colors[i]:
                    # self.cross_line(xmin, ymin, xmax, ymax, c.RED)
                    self.rect_detection_box(c.RED, xmin, ymin, xmax, ymax)
                    self.status_display("high".upper(), xmin, ymin, font=c.ORANGE, background=c.RED)
                    self.high_counter += 1
                else:
                    # self.cross_line(xmin, ymin, xmax, ymax, c.BLACK)
                    self.rect_detection_box(c.BLACK, xmin, ymin, xmax, ymax)
                    self.status_display("low".upper(), xmin, ymin, font=c.GREEN, background=c.BLACK)
                    self.low_counter += 1
                    
            if c.DASHBOARD_FLAG:
                self.generate_chart()
                self.dashboard_display()
            else: pass

            self.fps = self.generate_fps()
            self.information_display()
            self.frame_counter += 1
            self.log_time = datetime.datetime.now().strftime("%d-%m-%Y %I:%M:%S%p")
            self.generate_logging()

            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(self.window_name, self.frame)
             
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    def __del__(self) -> None:
        self.video.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = App()
    if c.LOGGING:
        print(app)