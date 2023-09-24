from utils.fcos_det_runner import FcosDetRunner
import argparse
import os
import sys
import json
import yaml
from tqdm import tqdm

def inference_dataset(data_dir, params_dict):

    img_list = os.listdir(data_dir)
    input_shape = [params_dict['input_h'], params_dict['input_w']]
    results = []
    PD = FcosDetRunner(params_dict['checkpoint'], input_shape=input_shape, max_objects=params_dict['max_objects'], score_thres=params_dict['score_thres'],
                       iou_thres=params_dict['iou_thres'], e2e_coco=params_dict['e2e_coco'])
    for img_name in tqdm(img_list):
        if img_name.split('.')[-1] not in ['png', 'jpg']:
            continue
        img_path = os.path.join(data_dir, img_name)
        bboxes = PD.run(img_path)
        results.append({'img_path': img_path, 'bbox': bboxes } )
    return results

        
def parse_args(args):
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='e2e inference script for inference an object detection network.')
    parser.add_argument('--img-path', type=str, help='Path to the dataset directory.')
    parser.add_argument('--params',type=str, help='Path to the init params file.')
    parser.add_argument('--save-path', type=str, help='Path to save output in json.')
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi). (-1 for cpu)',type=int,default=-1)

    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)           

def main(args = None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]

    args = parse_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Inference
    with open(args.params) as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    print('init params: ', params_dict)
    preds = inference_dataset(args.img_path, params_dict)

    with open(args.save_path, 'w') as fp:
        json.dump(preds, fp)

if __name__ == '__main__':
    main()