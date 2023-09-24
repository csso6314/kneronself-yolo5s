from utils.fcos_det_runner import FcosDetRunner
import argparse
import os
import sys
import cv2
import csv
import json
import numpy as np

def inference(model_path, img_path, input_shape=(416,416), max_objects=100, score_thres= 0.5, iou_thres = 0.5, nms = True, verbose=True):
    
    PD = FcosDetRunner(model_path,input_shape=input_shape,max_objects = max_objects, score_thres=score_thres, iou_thres = iou_thres, nms = nms)
    bboxes = PD.run(img_path)
    if verbose:
        print(bboxes)
 
    return bboxes

def get_id_class_dict(class_id_path):
    file_type = class_id_path.split('.')[-1]
    if file_type == 'csv':
        with open(class_id_path, mode='r') as infile:
            reader = csv.reader(infile)
            id_class_dict = {rows[1]:rows[0] for rows in reader}
            try:
                key_id = int(list(id_class_dict.keys())[0])
            except:
                id_class_dict = dict([(value, key) for key, value in id_class_dict.items()])
                
    elif file_type == 'json':
        with open(class_id_path) as json_file:
            id_class_dict = json.load(json_file)
            try:
                key_id = int(list(id_class_dict.keys())[0])
            except:
                id_class_dict = dict([(value, key) for key, value in id_class_dict.items()])
                
    else:
        print('Unsupported file type, exiting...')
        exit()
    return id_class_dict

def draw(img_path, bboxes, save_path = None, class_id_path = None):
    
    
    if class_id_path is not None:
        id_class_dict = get_id_class_dict(class_id_path)

    img = cv2.imread(img_path)
    for bbox in bboxes:
        l,t,w,h,score,class_id=bbox
        if class_id_path is not None:
            class_id = id_class_dict[str(int(class_id))]
        img = cv2.rectangle(img,(int(l),int(t)),(int(l+w),int(t+h)),(0, 255, 0),6)
        text = "{}".format(class_id) + "  {}".format(np.round(score, 3))
        img = cv2.putText(img, text, (int(l), int(t)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if save_path is None:
        save_path = img_path
    cv2.imwrite(save_path, img)
        
def parse_args(args):
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Simple inference script for inference an object detection network.')
    parser.add_argument('--img-path', type=str, help='Path to the image.')
    parser.add_argument('--class-id-path', help='Path to the class id mapping file.', default=None)
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi). (-1 for cpu)',type=int,default=-1)
    parser.add_argument('--snapshot', help='Path to the pretrained models.', default=None)
    parser.add_argument('--input-shape', help='Input shape of the model.', nargs='+', type=int, default=[512, 512])
    parser.add_argument('--max-objects', help='The maximum number of objects in the image.', type=int, default=100)
    parser.add_argument('--score-thres', help='The score threshold of bounding boxes.',type=float, default=0.6)
    parser.add_argument('--iou-thres', help='The IOU threshold of bounding boxes.',type=float, default=0.5)
    parser.add_argument('--nms', help='Whether use nms',type=int, default=1)
    parser.add_argument('--save-path', help='Path to draw and save the image with bbox', default=None)
    parser.add_argument('--save-preds-path', help='Path to save the inference bboxes', default=None)

    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)           

def main(args = None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]

    args = parse_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Inference 
    preds = inference(args.snapshot, args.img_path, args.input_shape, args.max_objects, args.score_thres, args.iou_thres, args.nms)
    if args.save_preds_path is None:
        image_name = args.img_path.split('/')[-1]
        image_folder = os.path.dirname(args.img_path)
        args.save_preds_path = os.path.join(image_folder, image_name.split('.')[0]+'_preds.json')
    dic = {}
    dic['img_path']=args.img_path
    dic['0_0']=preds

    with open(args.save_preds_path, 'w') as fp:
        json.dump(dic, fp)
        
    if args.save_path is not None:
        draw(args.img_path, preds, args.save_path, args.class_id_path)
    
    
    return preds

if __name__ == '__main__':
    main()