import json
import csv
import os
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def json2csv(data_dir, json_name, csv_name):
    json_name = os.path.join(data_dir, json_name)
    f_json = open(json_name)
    data = json.load(f_json)
    f_json.close()
    
    image_name = data['imagePath']
    with open(csv_name, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        for info in data['shapes']:
            l,t,r,b = str(info['points'][0][0]), str(info['points'][0][1]), str(info['points'][1][0]), str(info['points'][1][1])
            row = [[image_name, l,t,r,b, info['label'] ]]
            csvwriter.writerows(row)
        
def create_csv(data_dir, save_dir):

    fields = ['img_id', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id']
    save_path = os.path.join(save_dir, 'img_info.csv')
    with open(save_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
            
    for file in os.listdir(data_dir):
        if file.split('.')[-1] == 'json':
            json2csv(data_dir, file, save_path)
            
def split_data(data_cvs_path, save_dir):

    input_file = pd.read_csv(data_cvs_path)
    ids = np.array(list(set(input_file['img_id'])))
    
    ids = np.array(ids)
    n = len(ids)
    
    train_idx = np.random.choice(n, int(n*0.8), replace=False)
    val_idx = np.setdiff1d(np.arange(n), train_idx)
    
    train_img, val_img = set(ids[train_idx]), set(ids[val_idx])
    
    mask = []
    mask_ = []
    for i in range(len(input_file['img_id'].values)):
        if input_file['img_id'].values[i] in train_img:
            mask.append(True)
            mask_.append(False)
        else:
            mask.append(False)
            mask_.append(True)

    train_csv = input_file[mask]
    val_csv = input_file[mask_]
    train_csv.to_csv(os.path.join(save_dir, 'train_info.csv'), sep=',',index=False)
    val_csv.to_csv(os.path.join(save_dir, 'val_info.csv'), sep=',',index=False)
    

def create_class_mapping(data_cvs_path, save_dir):

    input_file = pd.read_csv(data_cvs_path)
    ids = list(set(input_file['class_id']))
    ids.sort()

    save_path = os.path.join(save_dir,'class_id.csv')
        
    with open(save_path, 'w') as f:
        write = csv.writer(f, delimiter=",")
        for i, name in enumerate(ids):
            write.writerow([name,str(i)])

def prepare(anno_cvs_path, save_dir):
    input_file = pd.read_csv(anno_cvs_path, header=None)
    ids = np.array(list(set(input_file[5])))
    n = len(ids)

    train_idx = np.random.choice(n, int(n*0.8), replace=False)
    val_idx = np.setdiff1d(np.arange(n), train_idx)

    train_img, val_img = set(ids[train_idx]), set(ids[val_idx])
    data = {}  
    data['img_id'] = input_file[5]
    data['xmin'], data['ymin'] = input_file[1], input_file[2]
    data['xmax'], data['ymax'] = input_file[1]+input_file[3], input_file[2]+input_file[4]
    data['class_id'] = input_file[0]
    df = pd.DataFrame(data)
    
    mask = []
    mask_ = []
    for i in range(len(df['img_id'].values)):
        if df['img_id'].values[i] in train_img:
            mask.append(True)
            mask_.append(False)
        else:
            mask.append(False)
            mask_.append(True)

    train_csv = df[mask]
    val_csv = df[mask_]
    print('saving train info into ', os.path.join(save_dir, 'train_info.csv'))
    train_csv.to_csv(os.path.join(save_dir, 'train_info.csv'), sep=',',index=False)
    print('saving val info into ', os.path.join(save_dir, 'val_info.csv'))
    val_csv.to_csv(os.path.join(save_dir, 'val_info.csv'), sep=',',index=False)

    class_id = list(set(data['class_id']))
    class_id.sort()
    save_path = os.path.join(save_dir,'class_id.csv')
    print('saving class id mapping into ', os.path.join(save_dir, 'class_id.csv'))    
    with open(save_path, 'w') as f:
        write = csv.writer(f, delimiter=",")
        for i, name in enumerate(class_id):
            write.writerow([name,str(i)])

def prepare_txt(train_dir, id_mapping, trainset = True):
    save_dir = os.path.split(train_dir)
    if len(save_dir[1]) == 0:
        save_dir = os.path.split(save_dir[0])
    trainval = save_dir[1]
    save_dir = save_dir[0]
    
    par_dir = os.path.split(save_dir)[0]
    txt_path = os.path.join(par_dir, 'labels', trainval) 
    imgs_path = train_dir

    fields = ['img_id', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id']
    if trainset:
        save_path = os.path.join(save_dir, 'train_info.csv')
    else:
        save_path = os.path.join(save_dir, 'val_info.csv')
        
    with open(save_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        
    for txt_file in os.listdir(txt_path):
        if txt_file[0] == '.':
            continue
        txt_file_path = os.path.join(txt_path, txt_file)
        img_id = txt_file.split('.')[0]+'.jpg'
        img_path = os.path.join(imgs_path, img_id)
        image = plt.imread(img_path)
        try:
            h,w,_ = image.shape
        except:
            h,w = image.shape
        with open(txt_file_path, 'r') as fp:
            content = fp.readlines()
            with open(save_path, 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                
                for data in content:
                    class_id,cx,cy,cw,ch = data.split(' ')
                    class_id = id_mapping[int(class_id)]
                    cx,cy,cw,ch = float(cx)*w, float(cy)*h, float(cw)*w, float(ch)*h

                    l,t,r,b = str(cx-cw/2), str(cy-ch/2), str(cx+cw/2), str(cy+ch/2)
                    row = [[img_id, l,t,r,b, class_id ]]
                    csvwriter.writerows(row)
    return save_path
            
def parse_args(args):
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='preprocessing csv data')
    parser.add_argument('annopath', help='path to csv file downloaded from makesense.ai')
    return parser.parse_args(args)

def main(args = None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    save_dir = os.path.split(args.annopath)
    if len(save_dir[1]) == 0:
        save_dir = os.path.split(save_dir[0])
    save_dir = save_dir[0]
        
    '''
    create_csv(args.anno_path, save_dir)
    data_cvs_path = os.path.join(save_dir, 'img_info.csv')
    split_data(data_cvs_path, save_dir)
    create_class_mapping(data_cvs_path, save_dir)
    '''
    
    prepare(args.annopath, save_dir)
    
if __name__ == '__main__':
    main()
