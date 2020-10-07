<div align="center">
  <img src="images/SoDV-v0.2.png">
</div>

# Social Distance Violation Detection (SoDV) using pre-trained YOLOv3 model ![SoDV](https://img.shields.io/badge/Build-v1.1-blue) [![LICENSE](https://img.shields.io/badge/license-MIT-blue)](https://github.com/afiqharith/SocialDistanceDetector-SODV/blob/master/LICENSE) [![FKE](https://img.shields.io/badge/FKE-UiTM-purple)](https://fke.uitm.edu.my/) [![RMC](https://img.shields.io/badge/RMC-UiTM-purple)](https://rmc.uitm.edu.my/v2/)

The previous development of this project used MobileNet SSD pre-trained on MS-COCO as the person detection algorithm. After the [previous project](https://github.com/afiqharith/SocialDistancing-SafetyViolationROI-MobileNetSSD-FYP.git 'Build v1.0 passing') finished in July 2020, I decided to further improve the detection algorithm by changing from MobileNet SSD to YOLOv3 to increase the accuracy. The program uses OpenCV API for the image processing and utilizing the DNN module which solely tested on CPU. The system accuracy tested on Oxford Town Centre CCTV video-dataset (530 frames). This project was submitted to Malaysia Technology Expo (MTE) Special Edition 2020 under Faculty of Electrical Engineering of Universiti Teknologi MARA.
</br>

_💻 Install the dependencies on command line:_

```sh
$ pip3 install -r requirement.txt
```

_💻 To run the program on command line:_

```sh
$ python3 social-distance-yolo.py
```

</br>

### 🎬 Output example:

| ![outputimage](/images/TownCentre_new.gif) |
| ------------------------------------------ |


**Frame 10 to 500:**

| ![outputimage](/images/data/frame%2010.jpg) | ![outputimage](/images/data/frame%20100.jpg) | ![outputimage](/images/data/frame%20200.jpg) |
| :-----------------------------------------: | :------------------------------------------: | :------------------------------------------: |
|                **FRAME 10**                 |                **FRAME 100**                 |                **FRAME 200**                 |

| ![outputimage](/images/data/frame%20300.jpg) | ![outputimage](/images/data/frame%20400.jpg) | ![outputimage](/images/data/frame%20500.jpg) |
| :------------------------------------------: | :------------------------------------------: | :------------------------------------------: |
|                **FRAME 300**                 |                **FRAME 400**                 |                **FRAME 500**                 |

### 🎯 Accuracy for person detection:

| Dataset            | TP  | TN  | FP  | FN  | %    |
| ------------------ | --- | --- | --- | --- | ---- |
| Oxford Town Centre | 29  | 0   | 0   | 11  | 72.5 |

### 🎯 Accuracy for social distance violation detection:

| Dataset            | TP  | TN  | FP  | FN  | %   |
| ------------------ | --- | --- | --- | --- | --- |
| Oxford Town Centre | NA  | NA  | NA  | NA  | NA  |

### ⛔ Social distance violation detection using distance formula Drawback:

- No camera calibration for intrinsic parameter

### ⏳ Future upgrades:

| Features                                              | Status                                                                |
| ----------------------------------------------------- | --------------------------------------------------------------------- |
| Camera calibration for intrinsic parameter (distance) | ![STATUS](https://img.shields.io/badge/camera_calibration-TBD-orange) |
| Add facemask detection                                | ![STATUS](https://img.shields.io/badge/facemask_detection-TBD-orange) |

---

## Kindly check out below links:

**🎥 Output video** </br>
[![Youtube](https://img.shields.io/badge/Social_Distance_Violation_Detection-Youtube-red)](https://www.youtube.com/watch?v=zXBDvDaJLHA)

**👀 YOLO Pre-Trained Model** </br>
[![YOLO](https://img.shields.io/badge/YOLO-Darknet-yellow)](https://pjreddie.com/darknet/yolo/) [![Darknet](https://img.shields.io/badge/Darknet-GitHub-lightgrey)](https://github.com/pjreddie/darknet.git)

**📊 Dataset** </br>
MegaPixels: Origins, Ethics, and Privacy Implications of Publicly Available Face Recognition Image Datasets </br>
[![Oxford Town Centre CCTV video-dataset](https://img.shields.io/badge/Oxford_Town_Centre-URL-yellowgreen)](https://megapixels.cc/oxford_town_centre/)
</br>

### LICENSE

_This project is under MIT license, please look at [LICENSE](https://github.com/afiqharith/SocialDistanceDetector-SODV/blob/master/LICENSE)._
