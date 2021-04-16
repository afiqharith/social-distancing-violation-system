<div align="center">
  <img src="images/sodv.png">
</div>

# SODV: Social Distancing Violation System YOLO version 3

![SoDV](https://img.shields.io/badge/Build-v1.1-blue) [![LICENSE](https://img.shields.io/badge/license-MIT-blue)](https://github.com/afiqharith/SocialDistanceDetector-SODV/blob/master/LICENSE) [![FKE](https://img.shields.io/badge/FKE-UiTM-purple)](https://fke.uitm.edu.my/) [![RMC](https://img.shields.io/badge/RMC-UiTM-purple)](https://rmc.uitm.edu.my/)

This project is a social distancing violation detection system implemented using Python. The previous development of this project used MobileNet SSD pre-trained on MS-COCO as the person detection algorithm. After the [previous project](https://github.com/afiqharith/SocialDistancing-SafetyViolationROI-MobileNetSSD-FYP.git 'Build v1.0 passing') finished in July 2020, I decided to further improve the detection algorithm by changing from MobileNet SSD to YOLOv3 to increase the accuracy. The program uses OpenCV for the image processing and utilizing the DNN module which solely tested on CPU. The system accuracy tested on Oxford Town Centre CCTV video-dataset (266 frames). This project was submitted to Malaysia Technology Expo (MTE) 2020 Special Edition COVID-19 International Innovation Awards under Faculty of Electrical Engineering of Universiti Teknologi MARA.
</br>

### 1. Prerequisites and configurations

All the requirements can be installed via the command:

```sh
$ pip3 install -r requirements.txt
```

The default input is video located in videos file. To change the program to use camera stream as input, you need to change the configuration from `CAMERA = False` to `CAMERA = True`.

Note: All configurations can be changed in the **config.py** file.
</br>

### 2. Run project

Run:

```sh
$ python social_distancing_violation_system.py
```

### 3. Program output

| ![outputimage](/images/TownCentre_new.gif) |
| ------------------------------------------ |

**Output frame 10 to 250:**

| ![outputimage](/images/data/frame_10.jpg) | ![outputimage](/images/data/frame_50.jpg) | ![outputimage](/images/data/frame_100.jpg) |
| :---------------------------------------: | :---------------------------------------: | :----------------------------------------: |
|               **FRAME 10**                |               **FRAME 50**                |               **FRAME 100**                |

| ![outputimage](/images/data/frame_150.jpg) | ![outputimage](/images/data/frame_200.jpg) | ![outputimage](/images/data/frame_250.jpg) |
| :----------------------------------------: | :----------------------------------------: | :----------------------------------------: |
|               **FRAME 150**                |               **FRAME 200**                |               **FRAME 250**                |

### 4. Overall accuracies

| ![graph](/images/graph.png) |
| --------------------------- |

### 5. Accuracy for person detection

| Dataset            | TP  | TN  | FP  | FN  | %    |
| ------------------ | --- | --- | --- | --- | ---- |
| Oxford Town Centre | 29  | 0   | 0   | 11  | 72.5 |

### 6. Accuracy for social distance violation detection

| Dataset            | TP  | TN  | FP  | FN  | %    |
| ------------------ | --- | --- | --- | --- | ---- |
| Oxford Town Centre | 23  | 15  | 9   | 2   | 77.5 |

### 7. Project drawback

- No camera calibration for intrinsic parameter

### 8. Future upgrades

| Features                                              | Status                                                                |
| ----------------------------------------------------- | --------------------------------------------------------------------- |
| Camera calibration for intrinsic parameter (distance) | ![STATUS](https://img.shields.io/badge/camera_calibration-TBD-orange) |
| Integration with facemask detection                   | ![STATUS](https://img.shields.io/badge/facemask_detection-TBD-orange) |
| Integration with DeepSort                             | ![STATUS](https://img.shields.io/badge/DeepSort-TBD-orange)           |

### 9. References

**Previous project** </br>
[Person Detection for Social Distancing and Safety Violation Alert based on Segmented ROI](https://github.com/afiqharith/SocialDistancing-SafetyViolationROI-MobileNetSSD-FYP.git 'GitHub Repo')

**Output video** </br>
[![Youtube](https://img.shields.io/badge/Social_Distance_Violation_Detection-Youtube-red)](https://www.youtube.com/watch?v=zXBDvDaJLHA)

**YOLO Pre-Trained Model** </br>
[![YOLO](https://img.shields.io/badge/YOLO-Darknet-yellow)](https://pjreddie.com/darknet/yolo/) [![Darknet](https://img.shields.io/badge/Darknet-GitHub-lightgrey)](https://github.com/pjreddie/darknet.git)

**Dataset** </br>
MegaPixels: Origins, Ethics, and Privacy Implications of Publicly Available Face Recognition Image Datasets </br>
[![Oxford Town Centre CCTV video-dataset](https://img.shields.io/badge/Oxford_Town_Centre-URL-yellowgreen)](https://exposing.ai/oxford_town_centre/)
</br>

#### LICENSE

_This project is licensed under the terms of the [MIT license](https://github.com/afiqharith/SocialDistanceDetector-SODV/blob/master/LICENSE)._
