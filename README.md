<div align="center">
  <img src="images/SoDV.png">
</div>

# Social Distance Violation Detection (SoDV) using pre-trained YOLOv3 model ![SoDV](https://img.shields.io/badge/Build-v1.1-BLUE)

The initial idea of this project is to use MobileNet SSD with Caffe implementation as the person detection algorithm. After I've finished my Final Year Project [![FYP](https://img.shields.io/badge/Build-v1.0_pass-BRIGHTGREEN)](https://github.com/afiqharith/Social-Distancing-and-Safety-Violation-Alert-ROI-MobileNetSSD-FYP) in July 2020, I decided to further improve the detection algorithm by using YOLOv3 to increase the accuracy.
</br>

_💻 Install the dependencies on command line:_ </br>

```sh
$ pip3 install -r requirement.txt
```

_💻 To run the program on command line:_ </br>

```sh
$ python3 social-distance-yolo.py
```

</br>

### 🎬 Output example:

| ![outputimage](/images/TownCentre_new.gif) |
| ------------------------------------------ |


**At frame 225:** </br>
| ![outputimage](/images/data/frame%20225.jpg) |
| -------------------------------------------- |

### 🎯 Accuracy for person detection:

| Dataset            | TP  | TN  | FP  | FN  | %    |
| ------------------ | --- | --- | --- | --- | ---- |
| Oxford Town Centre | 29  | 0   | 0   | 11  | 72.5 |

### 🎯 Accuracy for social distance violation detection:

| Dataset            | TP  | TN  | FP  | FN  | %   |
| ------------------ | --- | --- | --- | --- | --- |
| Oxford Town Centre | n/a | n/a | n/a | n/a | n/a |

### ⛔ Social distance violation detection using distance formula Drawback:

- No camera calibration for intrinsic parameter

### ⏳ Future upgrades:

| Features                                              | Status                                                                |
| ----------------------------------------------------- | --------------------------------------------------------------------- |
| Camera calibration for intrinsic parameter (distance) | ![STATUS](https://img.shields.io/badge/camera_calibration-TBD-orange) |
| Add facemask detection                                | ![STATUS](https://img.shields.io/badge/facemask_detection-TBD-orange) |

---

## Kindly check out links:

**🎥 Output video** </br>
[![Youtube](https://img.shields.io/badge/Social_Distance_Violation_Detection-Youtube-RED)](https://www.youtube.com/watch?v=zXBDvDaJLHA)

**👀 YOLO Pre-Trained Model** </br>
[![YOLO](https://img.shields.io/badge/YOLO-Darknet-YELLOW)](https://pjreddie.com/darknet/yolo/) [![Darknet](https://img.shields.io/badge/Darknet-GitHub-lightgrey)](https://github.com/pjreddie/darknet.git)

**📊 Dataset** </br>
MegaPixels: Origins, Ethics, and Privacy Implications of Publicly Available Face Recognition Image Datasets </br>
[![Oxford TownCentre](https://img.shields.io/badge/Oxford_Town_Centre-URL-yellowgreen)](https://megapixels.cc/)
