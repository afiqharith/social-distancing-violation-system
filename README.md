<div align="center">
  <img src="images/SoDV-v0.2.png">
</div>

# SODV: Social Distancing Violation System using pre-trained YOLOv3 model

![SoDV](https://img.shields.io/badge/Build-v1.1-blue) [![LICENSE](https://img.shields.io/badge/license-MIT-blue)](https://github.com/afiqharith/SocialDistanceDetector-SODV/blob/master/LICENSE) [![FKE](https://img.shields.io/badge/FKE-UiTM-purple)](https://fke.uitm.edu.my/) [![RMC](https://img.shields.io/badge/RMC-UiTM-purple)](https://rmc.uitm.edu.my/v2/)

The previous development of this project used MobileNet SSD pre-trained on MS-COCO as the person detection algorithm. After the [previous project](https://github.com/afiqharith/SocialDistancing-SafetyViolationROI-MobileNetSSD-FYP.git 'Build v1.0 passing') finished in July 2020, I decided to further improve the detection algorithm by changing from MobileNet SSD to YOLOv3 to increase the accuracy. The program uses OpenCV API for the image processing and utilizing the DNN module which solely tested on CPU. The system accuracy tested on Oxford Town Centre CCTV video-dataset (266 frames). This project was submitted to Malaysia Technology Expo (MTE) 2020 Special Edition COVID-19 International Innovation Awards under Faculty of Electrical Engineering of Universiti Teknologi MARA.
</br>

### Prerequisites:

_Install the dependencies on command line:_

```sh
$ pip3 install -r requirements.txt
```

_To run the program on command line:_

```sh
$ python3 SODV-app.py
```

_Edit program configuration on **config.py**:_

- To use device's camera as program input, change `CAMERA = False` to `CAMERA = True`.
  </br>

### Program Output:

| ![outputimage](/images/TownCentre_new.gif) |
| ------------------------------------------ |

**Frame 10 to 250:**

| ![outputimage](/images/data/frame_10.jpg) | ![outputimage](/images/data/frame_50.jpg) | ![outputimage](/images/data/frame_100.jpg) |
| :---------------------------------------: | :---------------------------------------: | :----------------------------------------: |
|               **FRAME 10**                |               **FRAME 50**                |               **FRAME 100**                |

| ![outputimage](/images/data/frame_150.jpg) | ![outputimage](/images/data/frame_200.jpg) | ![outputimage](/images/data/frame_250.jpg) |
| :----------------------------------------: | :----------------------------------------: | :----------------------------------------: |
|               **FRAME 150**                |               **FRAME 200**                |               **FRAME 250**                |

### Overall accuracies:

| ![graph](/images/graph.png) |
| --------------------------- |

### Accuracy for person detection:

| Dataset            | TP  | TN  | FP  | FN  | %    |
| ------------------ | --- | --- | --- | --- | ---- |
| Oxford Town Centre | 29  | 0   | 0   | 11  | 72.5 |

### Accuracy for social distance violation detection:

| Dataset            | TP  | TN  | FP  | FN  | %    |
| ------------------ | --- | --- | --- | --- | ---- |
| Oxford Town Centre | 23  | 15  | 9   | 2   | 77.5 |

### Social distance violation detection using distance formula Drawback:

- No camera calibration for intrinsic parameter

### Future upgrades:

| Features                                              | Status                                                                |
| ----------------------------------------------------- | --------------------------------------------------------------------- |
| Camera calibration for intrinsic parameter (distance) | ![STATUS](https://img.shields.io/badge/camera_calibration-TBD-orange) |
| Integration with facemask detection                   | ![STATUS](https://img.shields.io/badge/facemask_detection-TBD-orange) |
| Integration with DeepSort                             | ![STATUS](https://img.shields.io/badge/DeepSort-TBD-orange)           |

### Kindly check out below links for references:

**📑 Previous project** </br>
[Person Detection for Social Distancing and Safety Violation Alert based on Segmented ROI](https://github.com/afiqharith/SocialDistancing-SafetyViolationROI-MobileNetSSD-FYP.git 'GitHub Repo')

**🎥 Output video** </br>
[![Youtube](https://img.shields.io/badge/Social_Distance_Violation_Detection-Youtube-red)](https://www.youtube.com/watch?v=zXBDvDaJLHA)

**👀 YOLO Pre-Trained Model** </br>
[![YOLO](https://img.shields.io/badge/YOLO-Darknet-yellow)](https://pjreddie.com/darknet/yolo/) [![Darknet](https://img.shields.io/badge/Darknet-GitHub-lightgrey)](https://github.com/pjreddie/darknet.git)

**📊 Dataset** </br>
MegaPixels: Origins, Ethics, and Privacy Implications of Publicly Available Face Recognition Image Datasets </br>
[![Oxford Town Centre CCTV video-dataset](https://img.shields.io/badge/Oxford_Town_Centre-URL-yellowgreen)](https://exposing.ai/oxford_town_centre/)
</br>

### LICENSE

_This project is under MIT license, please look at [LICENSE](https://github.com/afiqharith/SocialDistanceDetector-SODV/blob/master/LICENSE)._
