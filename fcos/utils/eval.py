import keras
import numpy as np
import os
import pickle
import json
import sys
import argparse
import progressbar
import collections
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv

current_path=os.getcwd()
sys.path.append(current_path)

from utils.compute_overlap import compute_overlap
from utils.visualization import draw_detections, draw_annotations

from utils.fcos_det_runner import FcosDetRunner

assert (callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

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


def _compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    Args:
        recall: The recall curve (list).
        precision: The precision curve (list).

    Returns:
        The average precision as computed in py-faster-rcnn.

    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def _get_detections_from_json(detection_path, num_classes):
    """
    Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_class_detections, 5]

    Args:
        generator: The generator used to run images through the model.
        model: The model to run on the images.
        score_threshold: The score confidence threshold to use.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save the images with visualized detections to.

    Returns:
        A list of lists containing the detections for each image in the generator.

    """
    dets = {}
    for file in os.listdir(detection_path):
        if file.split('.')[-1] != 'json':
            continue
        full_filename = os.path.join(detection_path, file)
        with open(full_filename,'r') as fi:
            dic = json.load(fi)
            img_name = dic['img_path'].split('/')[-1]
            dets[img_name ] = dic["0_0"] # {img_id: [[score1,label1], [score2,label2]]}
    dets = collections.OrderedDict(sorted(dets.items()))
    all_detections = [[None for i in range(num_classes)] for j in range(len(dets))]
  
    for i, key in progressbar.progressbar(enumerate(dets), prefix='Getting detection bounding boxes: '):
        bboxes = np.array(dets[key])
        bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
        # copy detections to all_detections
        for label in range(num_classes):
            all_detections[i][label] = bboxes[np.array(bboxes[:, -1]) == label, :-1].copy()

    return all_detections

def _get_detections_runner(generator, model_path, input_shape, score_threshold=0.05, max_detections=100, save_path=None):
    """
    Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_class_detections, 5]

    Args:
        generator: The generator used to run images through the model.
        model: The model to run on the images.
        score_threshold: The score confidence threshold to use.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save the result.

    Returns:
        A list of lists containing the detections for each image in the generator.

    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in
                      range(generator.size())]
    PD = FcosDetRunner(model_path,input_shape=input_shape,max_objects = max_detections, score_thres=score_threshold)
    dic = {}
    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image = generator.load_image(i)
        bboxes = PD.run(raw_image)
        if save_path is not None:
            try:
                image_name = generator.image_ids[i]
            except:
                image_name = generator.image_names[i]
            dic[image_name] = bboxes
  
        bboxes = np.array(bboxes)    
        if len(bboxes) > 0:
            bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
            bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue
            if len(bboxes) > 0:
                all_detections[i][label] = bboxes[bboxes[:, -1] == label, :-1]
            else:
                all_detections[i][label] = [[]]
                
    if save_path is not None:
        with open(save_path, 'w') as fp:
            json.dump(dic, fp)
        
    return all_detections

def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """
    Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_class_detections, 5]

    Args:
        generator: The generator used to run images through the model.
        model: The model to run on the images.
        score_threshold: The score confidence threshold to use.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save the images with visualized detections to.

    Returns:
        A list of lists containing the detections for each image in the generator.

    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in
                      range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image = generator.load_image(i)
        image = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        # (n, 4)
        image_boxes = boxes[0, indices[scores_sort], :]
        # (n, )
        image_scores = scores[scores_sort]
        # (n, )
        image_labels = labels[0, indices[scores_sort]]
        # (n, 6)
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name,
                            score_threshold=score_threshold)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

    return all_detections


def _get_annotations(generator):
    """
    Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_annotations[num_images][num_classes] = annotations[num_class_annotations, 5]

    Args:
        generator: The generator used to retrieve ground truth annotations.

    Returns:
        A list of lists containing the annotations for each image in the generator.

    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue
            
            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations

def _get_annotations_from_json(anno_path, num_classes):
    """
    Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_class_detections, 5]

    Args:
        generator: The generator used to run images through the model.
        model: The model to run on the images.
        score_threshold: The score confidence threshold to use.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save the images with visualized detections to.

    Returns:
        A list of lists containing the detections for each image in the generator.

    """

    with open(anno_path,'r') as fi:
        anno = json.load(fi)
        
    anno = collections.OrderedDict(sorted(anno.items()))
    
    all_detections = [[None for i in range(num_classes)] for j in range(len(anno))]

    for i, key in progressbar.progressbar(enumerate(anno), prefix='Getting detection bounding boxes: '):
        bboxes = np.array(anno[key])
        bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
        # copy detections to all_detections
        for label in range(num_classes):
            all_detections[i][label] = bboxes[np.array(bboxes[:, -1]) == label, :-1].copy()

    return all_detections

def evaluate(
        generator,
        model,
        iou_threshold=0.35,
        score_threshold=0.15,
        max_detections=100,
        save_path=None,
        epoch=0,
        input_shape=(512,512),
        runner = False
    
):
    """
    Evaluate a given dataset using a given model.

    Args:
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        iou_threshold: The threshold used to consider when a detection is positive or negative.
        score_threshold: The score confidence threshold to use for detections.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save images with visualized detections to.
        epoch: epoch index

    Returns:
        A dict mapping class names to mAP scores.

    """
    # gather all detections and annotations
    if runner:
        all_detections = _get_detections_runner(generator, model, input_shape, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
        all_annotations = _get_annotations(generator)
    else:
        all_detections = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
        all_annotations = _get_annotations(generator)
    average_precisions = {}

    # all_detections = pickle.load(open('fcos/all_detections_11.pkl', 'rb'))
    # all_annotations = pickle.load(open('fcos/all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('fcos/all_detections_{}.pkl'.format(epoch + 1), 'wb'))
    # pickle.dump(all_annotations, open('fcos/all_annotations_{}.pkl'.format(epoch + 1), 'wb'))

    # process detections and annotations
    for label in progressbar.progressbar(range(generator.num_classes()), prefix='Computing mAP: '):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue
                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions

def eval_coco(data_dir, annotations_path, model_path, set_name, input_shape, save_path,score_threshold=0.15,iou_threshold=0.35):
    import os
    current_path=os.getcwd()
    sys.path.append(current_path)
    from generators.coco import CocoGeneratorEval
    from utils.coco_eval import evaluate_coco_runner
    
    common_args = {
        'batch_size': 1
    }
 
    generator = CocoGeneratorEval(
            data_dir,
            annotations_path,
            set_name = set_name,
            shuffle_groups=False,
            **common_args
        )
    
    average_precisions = evaluate(generator, model_path, input_shape=input_shape, epoch=0, runner = True, save_path=save_path,score_threshold=score_threshold,iou_threshold=iou_threshold)
    logs= []
    # compute per class average precision
    total_instances = []
    precisions = []
    for label, (average_precision, num_annotations) in average_precisions.items():
        res = '{:.0f} instances of class '.format(num_annotations)+generator.label_to_name(label)+' with average precision: {:.4f}'.format(average_precision)
        logs.append(res)
        print('{:.0f} instances of class'.format(num_annotations), generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
        total_instances.append(num_annotations)
        precisions.append(average_precision)
    mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
    print('mAP: {:.4f}'.format(mean_ap))
    logs.append('mAP : {:.4f}'.format(mean_ap))
    
    #coco_eval_stats = evaluate_coco_runner(generator, model_path, input_shape)
    '''
    logs = {}
    coco_tag = ['AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                'AP @[ IoU=0.50      | area=   all | maxDets=100 ]',
                'AP @[ IoU=0.75      | area=   all | maxDets=100 ]',
                'AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                'AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                'AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
                'AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
                'AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
                'AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                'AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                'AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                'AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]']

    for index, result in enumerate(coco_eval_stats):
        logs[coco_tag[index]] = result  
    logs['mAP'] = coco_eval_stats[1]
    '''
    
    return logs

def eval_csv(data_dir, annotations_path, classes_path, model_path, input_shape, save_path,score_threshold=0.15,iou_threshold=0.35):
    import os
    current_path=os.getcwd()
    sys.path.append(current_path)
    from generators.csv_ import CSVGenerator

    common_args = {
        'batch_size': 1
    }
 
    generator = CSVGenerator(
            annotations_path,
            classes_path,
            shuffle_groups=False,
            base_dir=data_dir,
            **common_args
        )
    
    average_precisions = evaluate(generator, model_path, input_shape=input_shape, epoch=0, runner = True, save_path=save_path,score_threshold=score_threshold,iou_threshold=iou_threshold)
    
    logs= []
    # compute per class average precision
    total_instances = []
    precisions = []
    for label, (average_precision, num_annotations) in average_precisions.items():
        res = '{:.0f} instances of class '.format(num_annotations)+generator.label_to_name(label)+' with average precision: {:.4f}'.format(average_precision)
        logs.append(res)
        print('{:.0f} instances of class'.format(num_annotations), generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
        total_instances.append(num_annotations)
        precisions.append(average_precision)
    mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
    print('mAP: {:.4f}'.format(mean_ap))
    logs.append('mAP : {:.4f}'.format(mean_ap))
        
    return logs

def eval_e2e(detections_path, annotations_path, num_classes,score_threshold=0.15, iou_threshold=0.35):

    all_detections = _get_detections_from_json(detections_path, num_classes)
    
    with open(annotations_path,'r') as fi:
        anno = json.load(fi)
      
    anno = collections.OrderedDict(sorted(anno.items()))
    
    all_annotations = [[None for i in range(num_classes)] for j in range(len(anno))]

    for i, key in progressbar.progressbar(enumerate(anno), prefix='Getting annotations bounding boxes: '):
        bboxes = np.array(anno[key])
        bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
        # copy detections to all_detections
        for label in range(num_classes):
            all_annotations[i][label] = bboxes[np.array(bboxes[:, -1]) == label, :-1].copy()
    average_precisions = {}
    
    
    # process detections and annotations
    for label in progressbar.progressbar(range(num_classes), prefix='Computing mAP: '):

        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(anno)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])
                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue
                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
    
    logs= []
    # compute per class average precision
    total_instances = []
    precisions = []
    
    for label, (average_precision, num_annotations) in average_precisions.items():
        res = '{:.0f} instances of class '.format(num_annotations)+str(label)+' with average precision: {:.4f}'.format(average_precision)
        logs.append(res)
        print('{:.0f} instances of class'.format(num_annotations),label, 'with average precision: {:.4f}'.format(average_precision))
        total_instances.append(num_annotations)
        precisions.append(average_precision)
    mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
    print('mAP: {:.4f}'.format(mean_ap))
    logs.append('mAP : {:.4f}'.format(mean_ap))
        
    return logs

def parse_args(args):
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Simple evaluation script for evaluating a object detection network.')
    
    parser.add_argument('--data', help='Path to the data yaml file. (located under ./data/).')
    
    parser.add_argument('--e2e', help='e2e evaluation.', action='store_true')
    parser.add_argument('--detections-path', help='Path to predictions directory.', default=None)
    parser.add_argument('--annotations-path', help='Path to CSV file containing annotations for testing.', default=None)
    parser.add_argument('--num-classes', help='the number of classes.',type=int , default=None)

    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi). (-1 for cpu)',type=int,default=-1)
    parser.add_argument('--snapshot', help='Path to the pretrained models.', default=None)
    parser.add_argument('--input-shape', help='Input shape of the model.', nargs='+', type=int, default=[512, 512])
    parser.add_argument('--save-path', help='path to save detection results.', type=str, default=None)
    parser.add_argument('--conf-thres', help='score threshold.', type=float, default=0.1)
    parser.add_argument('--iou-thres', help='iou threshold.', type=float, default=0.35)

    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)       

def main(args = None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]

    args = parse_args(args)
    # get dataset information
    
    if not args.e2e:    
        with open(args.data) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
        args.dataset_type = data_dict["dataset_type"]
        if args.dataset_type == 'csv':
            args.data_root = data_dict["train"]
            args.data_dir = data_dict["val"]
            args.classes_path = data_dict["names"]
            args.annotations_path = prepare_txt(data_dict["train"], data_dict["names"], trainset = True)
            args.val_annotations_path = prepare_txt(data_dict["val"], data_dict["names"], trainset = False)
        elif args.dataset_type == 'coco':
            args.data_dir = data_dict["data_root"]
            args.set_name = data_dict["val_set_name"]
            args.annotations_path = data_dict["val_annotations_path"]
        else:
            print('Unsupported dataset type.')
            return 
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # get average_precisions
    if args.e2e:
        stats = eval_e2e(args.detections_path, args.annotations_path, args.num_classes,args.conf_thres, args.iou_thres)
    elif args.dataset_type == 'coco':
        stats = eval_coco(args.data_dir, args.annotations_path, args.snapshot, args.set_name, args.input_shape, args.save_path,args.conf_thres,args.iou_thres)
    else:
        stats = eval_csv(args.data_dir, args.annotations_path,  args.classes_path, args.snapshot,args.input_shape, args.save_path, args.conf_thres, args.iou_thres)
        
    with open('mAP_result.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % line for line in stats)
    

if __name__ == '__main__':
    main()
    
