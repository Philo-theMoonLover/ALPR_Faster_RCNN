# ALPR_Faster_RCNN
This repository implements Faster RCNN for Automatic License Plate Recognition (ALPR), which includes License Plate detection and Characters recognition.

# Trainning with custom data
You can run 2 python files: `license_plate_detection_fasterrcnn.py` for License Plate detection and `license_plate_OCR_fasterrcnn.py` for characters recognition.
The dataset is using COCO format. You can make a custom dataset with annotation file on Workspace [roboflow.com](https://roboflow.com) and change the dataset path to yours.

* Train Detection model: This model is FasterRCNN with a backbone of MobileNetV3-Large

	`python3 license_plate_detection_fasterrcnn.py`

* Train Recognition model: same as Detection model.

	`python3 license_plate_OCR_fasterrcnn.py`

Although object detection algorithms are not commonly used in OCR problems due to the number of characters, this repository can still use an object detection model (e.g. Faster RCNN). This is because the number of characters in the license plate is usually not many (maximum 9 for Vietnamese vehicle numbers) and its size compared to that plate is relatively large. Now the problem has become object detection with many classes (36 classes from 0 to 9 and from 'A' to 'Z').

# Testing

Run `test_image.py` to detect plates (results saved at `results/results_Plate`) and save the processed plate to 'results/crop_plates' folder for Characters recognition.

Run `test_OCR.py` to recognize characters (results saved at `results/results_OCR`).

Fully processes implements in file `alpr_image.py`.

You can download the models I trained [here](https://drive.google.com/drive/folders/1aBgkTKwyrxrC2zVStFpFMSlvnHM3Ibl2?usp=sharing)

