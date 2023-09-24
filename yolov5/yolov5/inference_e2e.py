import os
import sys
import argparse
import yaml
from tqdm import tqdm
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, help='Path to the dataset directory.')
    parser.add_argument('--params', type=str, help='Path to the init params file.')
    parser.add_argument('--save-path', type=str, help='Path to save output in json.')
    
    args = parser.parse_args()
    
    par_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    sys.path.append(par_path)
    sys.path.append(os.path.join(par_path, 'exporting') )

    from yolov5.yolov5_runner import Yolov5Runner

    with open(args.params) as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
        
    num_classes = params_dict['nc']
    input_w = params_dict['input_w']
    input_h = params_dict['input_h']
    grid20_path = params_dict['grid20_path']
    grid40_path = params_dict['grid40_path']
    grid80_path = params_dict['grid80_path']
    conf_thres = params_dict['conf_thres']
    iou_thres = params_dict['iou_thres']
    model_type = params_dict['model_type']
    e2e_coco = params_dict['e2e_coco']

    if model_type == 'onnx':
        yolov5_model = Yolov5Runner(model_path=params_dict['onnx_path'], yaml_path=params_dict['model_yaml_path'], grid20_path=grid20_path, grid40_path=grid40_path, grid80_path=grid80_path, num_classes=num_classes, imgsz_h=input_h, imgsz_w=input_w, conf_thres=conf_thres, iou_thres=iou_thres, top_k_num=3000, vanish_point=0.0, e2e_coco=e2e_coco)
    else:
        yolov5_model = Yolov5Runner(model_path=params_dict['pt_path'], yaml_path=params_dict['model_yaml_path'], grid20_path=grid20_path, grid40_path=grid40_path, grid80_path=grid80_path, num_classes=num_classes, imgsz_h=input_h, imgsz_w=input_w, conf_thres=conf_thres, iou_thres=iou_thres, top_k_num=3000, vanish_point=0.0, e2e_coco=e2e_coco)

    img_list = os.listdir(args.img_path)
    results = []
    for img_name in tqdm(img_list):
        if img_name.split('.')[-1] not in ['png', 'jpg']:
            continue
        img_path = os.path.join(args.img_path, img_name)
        if model_type == 'onnx':
            bboxes = yolov5_model.run_onnx(img_path)
        else:
            bboxes = yolov5_model.run(img_path)
        results.append({'img_path': img_path, 'bbox': bboxes } )
    with open(args.save_path, 'w') as fp:
        json.dump(results, fp)
