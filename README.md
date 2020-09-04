# Social Distance using pre-trained YOLOv3 model

The initial idea of this project is to use MobileNet SSD with Caffe implementation as the person detection algorithm. After I've finished my Final Year Project on July 2020, I decided to further improve the detection algorithm by using YOLOv3 to increase the accuracy

_Install the dependencies on command line:_

```sh
pip3 install -r requirement.txt
```

_To run the program on command line:_

```sh
python3 social-distance-yolo.py
```

**Social distance using distance formula Drawback:**

- No camera calibration for intrinsic parameter

**Future upgrade:**

- [ ] Camera calibration for intrinsic parameter (distance)
- [ ] Add facemask detection

---

## Kindly check out below URL:

### 1. Output video

_Youtube:_ [Person Detection for Social Distance](https://youtu.be/zXBDvDaJLHA)

### 2. YOLO Pre-Trained Model

_Object detection model:_ [YOLO](https://pjreddie.com/darknet/yolo/)

_CC:_ [Darknet](https://pjreddie.com/)

### 3. Dataset

_Dataset from Oxford TownCentre:_ <https://megapixels.cc/>; MegaPixels: Origins, Ethics, and Privacy Implications of Publicly Available Face Recognition Image Datasets
