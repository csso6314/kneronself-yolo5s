#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys
import warnings
import yaml

import keras
import keras.preprocessing.image
import tensorflow as tf
from datetime import date

import losses
from models_bd.fcos import fcos
from callbacks import create_callbacks

from generators import create_generators
from utils.anchors import make_shapes_callback
from utils.anchors import AnchorParameters
from utils.config import read_config_file, parse_anchor_parameters
from utils.keras_version import check_keras_version
from utils.model import freeze as freeze_model
from utils.transform import random_transform_generator
from utils.image import random_visual_effect_generator
from csv_preprocess import prepare_txt

def get_session():
    """
    Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    return parsed_args


def parse_args(args):
    """
    Parse the arguments.
    """
    today = str(date.today())
    parser = argparse.ArgumentParser(description='Simple training script for training a FCOS network.')
    
    parser.add_argument('--data', help='Path to the data yaml file. (located under ./data/).')
    
    parser.add_argument('--snapshot', help='Resume training from a snapshot.')
    parser.add_argument('--backbone', help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--fpn', help='fpn model', default='simple', type=str)
    parser.add_argument('--reg-func', help='regression func', default='linear', type=str)
    parser.add_argument('--stage', help='num of stage', default=3, type=int)
    parser.add_argument('--head-type', help='head type', default='simple', type=str)
    parser.add_argument('--centerness-pos', help='centerness branch pos', default='reg', type=str)
    parser.add_argument('--batch-size', help='Size of the batches.', default=4, type=int)
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=100)
    parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=5000)
    parser.add_argument('--lr', help='Learning rate.', type=float, default=1e-4)
    parser.add_argument('--snapshot-path',
                        help='Path to store snapshots of models during training (defaults to \'snapshots\')',
                        default='snapshots/{}'.format(today))
    
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    
    parser.add_argument('--input-size', help='Rescale the image if the largest side is larger than max_side.',
                        type=int, default=512)
    parser.add_argument('--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss',
                        action='store_true')

    print(vars(parser.parse_args(args)))
    return check_args(parser.parse_args(args))


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    # get dataset information
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
        
    args.dataset_type = data_dict.get("dataset_type", 'csv')

    if args.dataset_type == 'csv':
        args.data_root = data_dict["train"]
        args.data_root_val = data_dict["val"]
        args.classes_path = data_dict["names"]
        args.annotations_path = prepare_txt(data_dict["train"], data_dict["names"], trainset = True)
        args.val_annotations_path = prepare_txt(data_dict["val"], data_dict["names"], trainset = False)
    elif args.dataset_type == 'coco':
        args.data_root = data_dict["data_root"]
    elif args.dataset_type == 'pascal':
        args.data_root = data_dict["data_root"]
        args.annotations_path = data_dict["train"]
        args.val_annotations_path = data_dict["val"]
    else:
        print('Unsupported dataset type.')
        return 
        
        
    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    n_stage = args.stage
    if n_stage==5:
        anchor_param = AnchorParameters.default
    elif n_stage==3:
        anchor_param = AnchorParameters(strides=[8, 16, 32],
                                        interest_sizes=[
                                            [-1, 64],
                                            [64, 128],
                                            [128, 1e8],
                                        ])
    else:
        assert 0

    # create the generators
    if args.dataset_type == 'csv':
        train_generator, validation_generator = create_generators(input_size=args.input_size,
                                                                  data_root=args.data_root,
                                                                  data_root_val=args.data_root_val,
                                                                  annotations_path=args.annotations_path,
                                                                  val_annotations_path=args.val_annotations_path,
                                                                  dataset_type=args.dataset_type,
                                                                  batch_size=args.batch_size,
                                                                  anchor_param=anchor_param,
                                                                  classes_path=args.classes_path)
    else:
        train_generator, validation_generator = create_generators(input_size=args.input_size,
                                                                  data_root=args.data_root,
                                                                  annotations_path=args.annotations_path,
                                                                  val_annotations_path=args.val_annotations_path,
                                                                  dataset_type=args.dataset_type,
                                                                  batch_size=args.batch_size,
                                                                  anchor_param=anchor_param)
    model, prediction_model, debug_model = fcos(backbone=args.backbone, 
                                                num_classes=train_generator.num_classes(),
                                                input_size=args.input_size,
                                                weights=args.snapshot,
                                                freeze_backbone=args.freeze_backbone,
                                                fpn_type=args.fpn,
                                                n_stage=n_stage,
                                                mapping_func=args.reg_func,
                                                head_type=args.head_type,
                                                centerness_pos=args.centerness_pos)


    training_model = model
    training_model.compile(
            loss={
                'regression': losses.giou,
                'classification': losses.focal(),
                'centerness': losses.bce(),
            },
            optimizer=keras.optimizers.Adam(lr=args.lr),
           loss_weights={'regression':2, 'classification':1, 'centerness':0.7}
        )
    

    # create the callbacks
    callbacks = create_callbacks(
        debug_model,
        prediction_model,
        validation_generator,
        args.snapshot_path,
        dataset_type=args.dataset_type,
        backbone=args.backbone,
        fpn=args.fpn,
        n_stage=n_stage
    )

    if not args.compute_val_loss:
        validation_generator = None
    
    # start training
    trained_model = training_model.fit_generator(
        generator=train_generator,
        initial_epoch=0,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator
    )
    

if __name__ == '__main__':
    main()
