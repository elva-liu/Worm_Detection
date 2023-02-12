# Worm_Detection
This program uses the YOLO algorithm to detect  worms under a microscope stream in real time. Then, a shutter mechanism will be triggered once a worm is detected, allowing UV light to pass through and polymerize the portion of the stream that contains the worm.

1. YOLO_model_worm_detect: is the worm detection model trained by YOLOv5 algorithm 
2. worm_best.pt: is the the weight file of the worm detection model
3. camera_detection: apply the object detection model that has been trained to operate in real-time with a webcam. 
                     (origianl source:https://github.com/niconielsen32/ComputerVision/blob/master/yoloCustomObjectDetection.py )
4. Shutte.py: was provied by the shutter manufactuer, Sutter Instrument (https://www.sutter.com)
5. Hamamatsu: this folder is provide by the camera manufacturer Hamamatsu (https://www.hamamatsu.com) 

