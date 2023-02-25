import scipy.spatial.distance as dst
import matplotlib.pyplot as plt
from common import BoundingBox
from config import Config
import numpy as np
import threading
import logging
import time
import cv2
import os

if Config.LOGGING:
    logging.basicConfig(filename=Config.LOG_PATH, level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s', filemode="a")
else:
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s',level=logging.DEBUG)

class Model:
    def __init__(self, weight_path, config_path, coconames_path, cuda_enabled = False) -> None:
        """
        Set model object instance attribute
        """
        self.weight_path = weight_path
        self.config_path = config_path
        self.coconames_path = coconames_path
        self.cuda_enabled = cuda_enabled

        """
        Initialize model object
        """
        self.object_label = self.get_object_label_list()
        self.network = self.get_dnn_network_layer()
        self.set_preferable_backend_and_target()

    def get_object_label_list(self) -> list:
        with open(self.coconames_path, "r") as f:
            label_list = [line.strip() for line in f.readlines()]
        return label_list

    def get_dnn_network_layer(self):
        return cv2.dnn.readNet(self.weight_path, self.config_path)

    def get_network_layers_name(self):
        pre_layer_names = self.network.getLayerNames()
        return [pre_layer_names[i[0] - 1] for i in self.network.getUnconnectedOutLayers()]

    def set_preferable_backend_and_target(self):
        if self.cuda_enabled:
            self.get_dnn_network_layer().setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.get_dnn_network_layer().setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            logging.info(f"Setting up preferable backend and target to CUDA")
        else:
            self.get_dnn_network_layer().setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.get_dnn_network_layer().setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logging.info(f"Setting up preferable backend and target to CPU")

class MainCtrl:
    def __init__(self) -> None:
        logging.info(f"Loading video source")
        self.video = cv2.VideoCapture(os.path.join(os.getcwd(), Config.FOLDERNAME, Config.VIDEONAME))
        self.model = Model(Config.WEIGHT_ABS_PATH, Config.CFG_ABS_PATH, Config.LABELS_ABS_PATH, cuda_enabled=False)
        self.start_frame_time = 0

    def draw_object_bounding_box(self, frame, bbox: BoundingBox):

        if bbox.is_violate == BoundingBox.VIOLATE:
            color = Config.RED
        else:
            color = Config.BLACK

        cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), color, 1)
        logging.debug("min = ({0}, {1}) max = ({2}, {3}) IsViolate = {4}".format(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, bbox.is_violate))

    def draw_object_violation_status(self, frame, bbox: BoundingBox):
        if bbox.is_violate == BoundingBox.VIOLATE:
            text = "high".upper()
            font_color = Config.ORANGE
            backgroud_color = Config.RED
        else:
            text = "low".upper()
            font_color = Config.GREEN
            backgroud_color = Config.BLACK

        label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        ylabel = max(bbox.ymin, label_size[1])
        cv2.rectangle(
            frame, 
            (bbox.xmin, ylabel - label_size[1]), 
            (bbox.xmin + label_size[0], bbox.ymin + base_line), 
            backgroud_color, cv2.FILLED)

        cv2.putText(
            frame, 
            text, 
            (bbox.xmin, bbox.ymin), 
            cv2.FONT_HERSHEY_DUPLEX, 
            0.5, 
            font_color, 
            1, 
            cv2.LINE_AA)

    def draw_pairwise_bbox_distance_line(self, frame, bbox_new: BoundingBox, bbox_old: BoundingBox):
        cv2.line(
            frame, 
            (bbox_new.get_groundplane_center_point()[0], bbox_new.get_groundplane_center_point()[1]), 
            (bbox_old.get_groundplane_center_point()[0], bbox_old.get_groundplane_center_point()[1]), 
            Config.YELLOW, 
            1, 
            cv2.LINE_AA)

        cv2.circle(
            frame, 
            (bbox_new.get_groundplane_center_point()[0], bbox_new.get_groundplane_center_point()[1]), 
            3, 
            Config.ORANGE, 
            -1, 
            cv2.LINE_AA)

        cv2.circle(
            frame, 
            (bbox_old.get_groundplane_center_point()[0], bbox_old.get_groundplane_center_point()[1]), 
            3, 
            Config.ORANGE, 
            -1, 
            cv2.LINE_AA)

    def draw_current_frame_legend(self, frame, bbox_list: BoundingBox):
        cv2.rectangle(frame, (13, 5), (250, 30), Config.BLACK, cv2.FILLED)
        cv2.putText(frame, "", (28, 24), cv2.FONT_HERSHEY_DUPLEX, 0.5, Config.WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, "{0}fps".format(self.get_current_fps()), (200, 24), cv2.FONT_HERSHEY_DUPLEX, 0.5, Config.GREEN, 1, cv2.LINE_AA)

        high_count = 0
        low_count = 0
        for i, bbox in enumerate(bbox_list):
            if bbox.is_violate == BoundingBox.VIOLATE:
                high_count += 1
            else:
                low_count += 1

        cv2.rectangle(frame, (13, 30), (250, 80), Config.GREY, cv2.FILLED)
        LINE = "--"
        HIGHRISK_TEXT = "HIGH RISK: {0} people".format(high_count)
        cv2.putText(frame, LINE, (28, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, Config.RED, 2, cv2.LINE_AA)
        cv2.putText(frame, HIGHRISK_TEXT, (60, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, Config.BLUE, 1, cv2.LINE_AA)

        LOWRISK_TEXT = "LOW RISK: {0} people".format(low_count)
        cv2.putText(frame, LINE, (28, 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, Config.BLACK, 2, cv2.LINE_AA)
        cv2.putText(frame, LOWRISK_TEXT, (60, 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, Config.BLUE, 1, cv2.LINE_AA)

    def get_current_fps(self):
        current_frame_time = time.perf_counter()
        fps = 1 / (current_frame_time - self.start_frame_time)
        self.start_frame_time = current_frame_time
        return int(fps)

    def init_chart_image(self, bbox_list: BoundingBox):
        high_count = 0
        low_count = 0
        for i, bbox in enumerate(bbox_list):
            if bbox.is_violate == BoundingBox.VIOLATE:
                high_count += 1
            else:
                low_count += 1

        _, ax = plt.subplots(figsize=(4,4))
        ax.pie([high_count, low_count], labels = ["High risk: {0}".format(high_count), "Low risk: {0}".format(low_count)], colors=[Config.RED_DB, Config.GREEN_DB])
        ax.legend()
        plt.savefig(Config.DASHBOARD_PATH, transparent=True, dpi=700)
        plt.close()

    def show_dashboard(self, bbox_list: BoundingBox):
        self.init_chart_image(bbox_list)
        dashboard = cv2.imread(Config.DASHBOARD_PATH)
        cv2.namedWindow("SODV Dashboard", cv2.WINDOW_NORMAL)
        cv2.imshow("SODV Dashboard", dashboard)

    def main(self):

        while (self.video.isOpened()):
            
            b_frame_exist, frame = self.video.read()

            # TBD
            if Config.THREAD:
                try:
                    self.thread_1 = threading.Thread(target=self.video.read)
                    self.thread_1.daemon = True
                    self.thread_1.start()
                except RuntimeError as err:
                    logging.error(err)
            else:
                pass

            if b_frame_exist:
                '''
                Resize the frame for the object prediction
                '''
                frame_resized = cv2.resize(frame, (416, 416))
            else:
                break

            blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.model.network.setInput(blob)
            layer_output_list = self.model.network.forward(self.model.get_network_layers_name())

            clustered_bounding_box_list = list()
            confidence_list = list()
            current_frame_height, current_frame_width, _ = frame.shape

            for _, layer_output in enumerate(layer_output_list):
                for _, detection in enumerate(layer_output):
                    score_list = detection[5:]
                    object_class_num = np.argmax(score_list)
                    object_confidence = score_list[object_class_num]

                    if object_class_num != 0x00:
                        continue

                    if object_confidence > Config.MIN_CONFIDENCE:

                        clustered_object_bounding_box_center_axis_x = int(detection[0] * current_frame_width)
                        clustered_object_bounding_box_center_axis_y = int(detection[1] * current_frame_height)

                        clustered_object_bounding_box_width = int(detection[2] * current_frame_width)
                        clustered_object_bounding_box_height = int(detection[3] * current_frame_height)

                        clustered_object_bounding_box_minimum_axis_x = int(clustered_object_bounding_box_center_axis_x - clustered_object_bounding_box_width / 2)
                        clustered_object_bounding_box_minimum_axis_y = int(clustered_object_bounding_box_center_axis_y - clustered_object_bounding_box_height / 2)

                        clustered_bounding_box_list.append(
                            [clustered_object_bounding_box_minimum_axis_x, 
                             clustered_object_bounding_box_minimum_axis_y, 
                             clustered_object_bounding_box_width, 
                             clustered_object_bounding_box_height])

                        confidence_list.append(float(object_confidence))

            """
             Apply non-max suppression (NMS) for the clustered detected object class bbox
            """
            current_frame_detected_object_0x00_bounding_box_list = list()
            post_frame_detected_object_0x00_bounding_box_list = list()
            nms_index_list = cv2.dnn.NMSBoxes(clustered_bounding_box_list, confidence_list, Config.MIN_CONFIDENCE, Config.NMS_THRESHOLD)

            for i, clustered_bounding_box in enumerate(clustered_bounding_box_list):

                if i in nms_index_list:
                    x, y, width, height = clustered_bounding_box
                    bounding_box_new = BoundingBox(x, y, width, height, BoundingBox.NON_VIOLATE)

                    current_frame_detected_object_0x00_bounding_box_list.append(bounding_box_new)

                    for j, bounding_box_old in enumerate(post_frame_detected_object_0x00_bounding_box_list):

                        bounding_box_new_ground_plane_center_point = bounding_box_new.get_groundplane_center_point()
                        bounding_box_old_ground_plane_center_point = bounding_box_old.get_groundplane_center_point()

                        if dst.euclidean(
                            list([bounding_box_new_ground_plane_center_point[0], bounding_box_new_ground_plane_center_point[1]]),
                            list([bounding_box_old_ground_plane_center_point[0], bounding_box_old_ground_plane_center_point[1]])) <= Config.DISTANCE:

                            bounding_box_new.is_violate = BoundingBox.VIOLATE
                            bounding_box_old.is_violate = BoundingBox.VIOLATE
                            self.draw_pairwise_bbox_distance_line(frame, bounding_box_new, bounding_box_old)
                            break
                    post_frame_detected_object_0x00_bounding_box_list.append(bounding_box_new)

            for i, bounding_box in enumerate(current_frame_detected_object_0x00_bounding_box_list):
                
                self.draw_object_bounding_box(frame, bounding_box)
                self.draw_object_violation_status(frame, bounding_box)
            
            if Config.DASHBOARD_FLAG:
                self.show_dashboard(current_frame_detected_object_0x00_bounding_box_list)

            self.draw_current_frame_legend(frame, current_frame_detected_object_0x00_bounding_box_list)

            window_text = "SODV: Social Distancing Violation System"
            cv2.namedWindow(window_text, cv2.WINDOW_NORMAL)
            cv2.imshow(window_text, frame)
             
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(window_text, cv2.WND_PROP_VISIBLE) < 1:
                break

    def __del__(self) -> None:
        self.video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = MainCtrl()
    app.main()