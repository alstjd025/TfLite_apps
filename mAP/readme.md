**extra process for getting mAP in YOLOv4-tiny**

#Precondition 
git clone https://github.com/easyhardhoon/mAP_TF.git

#About get_mAP_app
1. load input images from mAP_TF/input/images-optional
2. prepare each image and YOLOv4-tiny & execute TFLITE(source : FBF-TF)  
3. parse output tensor and move them to mAP_Tf/input/detection-results 

#Evaluate mAP
1. cd ../../mAP_TF 
2. run main.py (evaluate mAP code)

#Results
*mAP : 34.21* by COCO dataset (300 images), on TFLITE with CPU only 


