# vanilla_gpu w & w/o workload
#./scheduler yg & ../yolo_apps/case_study/yolo_case_study ../models/yolov4-tiny-416_ieie.tflite ../models/yolov4-tiny-416_ieie.tflite COCO416 0 0 0

# ours
#./scheduler yb & ../yolo_apps/case_study/yolo_case_study ../models/yolov4-tiny-416_ieie.tflite ../models/yolov4-tiny-416_ieie.tflite COCO416 0 0 0

# ours + with gpu workload [should run ./dummy_workload 60 0 1]
./scheduler yb_candi & ../yolo_apps/case_study/yolo_case_study ../models/yolov4-tiny-416_ieie.tflite ../models/yolov4-tiny-416_ieie.tflite COCO416 0 0 0
