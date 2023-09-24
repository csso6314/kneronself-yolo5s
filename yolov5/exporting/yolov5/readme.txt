(1)yolov5_app.py for ploting moldel inference results
cd applications
python yolov5_app.py

(2)yolov5_evaluation.py for evaluating moldel mAP at hw_repo
cd applications
python yolov5_evaluation.py

#mAP @ yolov5s_v2_op9_sig_batch1_input05_640x640_nearest_convert.onnx with(upsampling rearest)
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.346
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.533
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.372
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.196
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.442
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.279
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.456
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.503
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.557
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.618

(3)yolov5_compare_pth_onnx.py for comparing the results of pytorch moldel and onnx model
cd applications
python yolov5_compare_pth_onnx.py

(4)v2 model is in the below link.
10.200.210.221:/mnt/models/Object_models/YOLOv5/yolov5s_v2_state_dict_input05.pt
10.200.210.221:/mnt/models/Object_models/YOLOv5/yolov5s_v2_op9_sig_batch1_input05_640x640_nearest_convert.onnx

(5)the parameters setting
(5.1)In order to get high mAP in coco val2017, please use 
101620_yolov5_init_params.json
{
    "model_path": "/mnt/models/Object_models/YOLOv5/yolov5s_v2_state_dict_input05.pt",
    "grid20_path": "/mnt/models/Object_models/YOLOv5/20_640x640.npy",
    "grid40_path": "/mnt/models/Object_models/YOLOv5/40_640x640.npy",
    "grid80_path": "/mnt/models/Object_models/YOLOv5/80_640x640.npy",
    "num_classes": 80,
    "imgsz_h": 640,
	"imgsz_w": 640,
	"conf_thres": 0.001,
	"iou_thres": 0.65,
    "top_k_num": 3000
}


(5.2)For video usage scenarios, please use
102320_yolov5_init_params.json
{
    "model_path": "/mnt/models/Object_models/YOLOv5/yolov5s_v2_state_dict_input05.pt",
    "grid20_path": "/mnt/models/Object_models/YOLOv5/20_640x352.npy",
    "grid40_path": "/mnt/models/Object_models/YOLOv5/40_640x352.npy",
    "grid80_path": "/mnt/models/Object_models/YOLOv5/80_640x352.npy",
    "num_classes": 80,
    "imgsz_h": 352,
	"imgsz_w": 640,
	"conf_thres": 0.3,
	"iou_thres": 0.5,
    "top_k_num": 3000
}

(5.3)The differences of above setting are
(5.3.1) Video uses input (640w*352h) to run faster.
Coco has high or flat wide images, so it is better to use input (640w*640h) 
 
(5.3.2) Using the yolov5 official website setting test coco val2017, the confidence setting is low "conf_thres": 0.001, and the iou setting of NMS is high "iou_thres": 0.65, which gets a better mAP.
But running video needs to be set to "conf_thres": 0.3, so that there are not too many false positives, and the iou setting of NMS "iou_thres": 0.5 is more friendly to close objects
