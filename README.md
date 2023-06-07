# Rail-Intrusion-Detection
# updated on June 6, 2023
Foreign object intrusion warning assessment on railroad based on YOLO v5 
#For training YOLO v5, you can visit https://github.com/ultralytics/yolov5 to learn and  use model learning and training individually. We have used yolov5m.pt for pre-training.
Procedure with our trained model and data：
1. Move the data folder to a location parallel to the yolov5 file.
2. Need to specify the path to data in the file of risk_detect.py. Other parameters are fixed, but can be customized according to https://github.com/ultralytics/yolov5.
3. Modify the code on line 198 which is in utils/dataloaders.py to make the order of the input images follow the number size.
4. Run the file of risk_detect.py
5.The results of the recognized images appear in the runs/detect path, and the calculation time is recorded in the data/time_record.txt file.
