import keras
from pycocotools.cocoeval import COCOeval
from utils.coco_eval import evaluate_coco

import numpy as np
import json
from tqdm import trange
import cv2


class CocoEval(keras.callbacks.Callback):
    """ Performs COCO evaluation on each epoch.
    """

    def __init__(self, generator, model, tensorboard=None, threshold=0.05):
        """ CocoEval callback intializer.

        Args
            generator   : The generator used for creating validation data.
            tensorboard : If given, the results will be written to tensorboard.
            threshold   : The score threshold to use.
        """
        self.generator = generator
        self.threshold = threshold
        self.tensorboard = tensorboard
        self.active_model = model

        super(CocoEval, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

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
        coco_eval_stats = evaluate_coco(self.generator, self.active_model, self.threshold)
        if coco_eval_stats is not None and self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            for index, result in enumerate(coco_eval_stats):
                summary_value = summary.value.add()
                summary_value.simple_value = result
                summary_value.tag = '{}. {}'.format(index + 1, coco_tag[index])
                self.tensorboard.writer.add_summary(summary, epoch)
                logs[coco_tag[index]] = result
        logs['mAP'] = coco_eval_stats[1]
