import os
import torch
import sys
import argparse
import yaml

def save_weight(num_classes): 
    from models.yolo import Model  
    num_classes = num_classes 
    device=torch.device('cpu')
    ckpt = torch.load(path, map_location=device)
    model = Model(yaml_path, nc=num_classes)
    ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(ckpt['model'])
    torch.save(model.state_dict(),pt_path,_use_new_zipfile_serialization=False)


if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/pretrained_paths_520.yaml', help='the path to pretrained model paths yaml file')
    args = parser.parse_args()
    
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
        
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input_w = data_dict['input_w']
    input_h = data_dict['input_h']
    grid_dir = data_dict['grid_dir']
    grid20_path = data_dict['grid20_path']
    grid40_path = data_dict['grid40_path']
    grid80_path = data_dict['grid80_path']
    path = data_dict['path']
    pt_path=data_dict['pt_path']
    yaml_path=data_dict['yaml_path']

    save_weight(data_dict['nc']) 





