# *OBJECT DETECTION USING MASK R-CNN PROJECT*
## ABSTRACT :
Object Detection and Instance Segmentation using Mask R-CNN is a Python-based computer vision system that leverages the Mask R-CNN algorithm to perform real-time object detection and instance segmentation on video frames. The system utilizes the OpenCV library for image processing and deep neural network operations.
  
Here I used a pre-trained model Mask R-CNN using Inception V2 backbone network to train the object_detection_classes_coco.txt dataset.This dataset provides a wide range of object categories, enabling the model to detect various objects such as people, cars, animals, and everyday items.

By calling readNetFromTensorflow() with weights(frozen_inference_graph.pb) and config (mask_rcnn_incepion_v2_coco_2018_01_28.pbtxt) function will load the model and crete a network.It will do the computer vision task such as image classification,object detectio and image segmentation.

By using the pretrained model we can do the object detection using images and videos.It will performs object detection by localizing and identifying objects within video frames. Each detected object is labeled with its class and bounded by a rectangular box (bounding box).

In this project it offers configurability through command-line arguments. Users can specify the input video file, the paths to the Mask R-CNN model files, and adjust parameters such as the confidence threshold for object detection and the segmentation threshold for masks.

Additional features have been incorporated such as inference time calculations and frames per second (FPS) calculations to evaluate the model's performance. The project is developed and executed on a Linux-based operating system, utilizing the power of the OpenCV library for video processing and visualization.

This project aims to enhance understanding of deep learning techniques for object detection, improve skills in model deployment on Linux systems, and gain hands-on experience with state-of-the-art computer vision algorithms.

## System Configuration:
- System OS : Ubuntu 20.04
- RAM : 15.5 GB
- Disk Space : 500.1 GB
- Processor : Intel Core i5-7300U
- Graphics : Mesa Intel HD Graphics 620
- IDE : Visual Studio Code
## Packages and LIbraries Required
- numpy
- argparse
- imutils
- time
- cv2
- os

# FPS and Inference Time examined in Mask R-CNN

--------------------------------------------------
|SI no|Resolution|Inference time|FPS             |
|-----|----------|--------------|----------------|
|1    |256*256   |2200-2350     |0.43,0.44,0.45  |
|2    |320*320   |2200-2400     |0.43,0.44       |
|3    |480*480   |2200-2600     |0.42,0.43       |
|4    |600*600   |2300-2450     |0.41,0.42       |
|5    |800*800   |2400-2600     |0.40,0.39       |
|6    |1024*1024 |2900-3050     |0.33,0.34       |
|7    |1200*1200 |2900-3000     |0.33,0.34,0.35  |
|8    |1440*1440 |3100-3300     |0.31,0.32       |
|9    |1600*1600 |3500-3650     |0.28,0.27       |
--------------------------------------------------
