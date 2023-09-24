from __future__ import absolute_import
from keras.layers import *
from keras.models import *
from keras.regularizers import *
from keras_resnet import models as resnet_models
import numpy as np
import keras
# from losses import loss, CTLoss, RGLoss, HMLoss
import tensorflow as tf
from models_bd.darknet import cspdarknet53
from models_bd.dla import DLA, DLA_seg
from models_bd.fpn import simple_fpn, fpn, bifpn, fpn_pan
from models_bd.resnet import base_model
from models_bd.utils import nms, topk, evaluate_batch_item, decode, decode_ltrb
from models_bd.utils import get_conv_opt, get_convdw_opt, get_bn_opt, Act

conv_option = get_conv_opt(bias=False)
conv_dw_option = get_convdw_opt()
bn_option = get_bn_opt()
act_type = 'relu'
option_head = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }
option_head.update(conv_option)
option_head['use_bias'] = True

def head_model(input, feature, stage):
    output = input
    for i in range(4):
        output = keras.layers.Conv2D(
            filters=feature,
            activation='relu',
            name='stage_{}_conv_{}'.format(stage,i),
            bias_initializer='zeros',
            **option_head
        )(output)
    #add normal
    return output

def head_models(input, feature, stage):
    output = input
    for i in range(1):
        output = keras.layers.Conv2D(
            filters=feature,
            kernel_size=1,
            activation='relu',
            name='stage_{}_conv_{}'.format(stage, i),
            bias_initializer='zeros',
        )(output)
    #add normal
    return output


def fcos(num_classes,
         backbone='resnet18',
         input_size=512,
         nms=True,
         freeze_bn=False,
         freeze_backbone=False,
         fpn_type='fpn',
         weights=None,
         mode=1,
         n_stage=5,
         mapping_func='exp',
         centerness_pos='reg',
         head_type='ori'
         ):
    #assert backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'dla34', 'yolov5','yolov5ss']
    assert backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'dla34', 'darknet53s','darknet53ss']
    # output_size = input_size // 4
    if input_size is None or isinstance(input_size, int):
        image_input = Input(shape=(input_size, input_size, 3))
    else:
        assert len(input_size) == 2
        image_input = Input(shape=input_size+(3,))

    alpha = 1.0
    feature_size = 256
    if backbone == 'resnet18':
        #backbone_m = resnet_models.ResNet18(image_input, include_top=False)
        
        backbone_m = base_model(image_input)
    elif backbone == 'resnet34':
        backbone_m = resnet_models.ResNet34(image_input, include_top=False)
    elif backbone == 'resnet50':
        backbone_m = resnet_models.ResNet50(image_input, include_top=False)
    elif backbone == 'resnet101':
        backbone_m = resnet_models.ResNet101(image_input, include_top=False)
    elif backbone == 'resnet152':
        backbone_m = resnet_models.ResNet152(image_input, include_top=False)
    elif backbone == 'dla34':
        backbone_m = DLA(image_input, n_layer=34, include_top=False)
    elif backbone == 'darknet53s':
        alpha = 1.0
        backbone_m = cspdarknet53(image_input, alpha=1.0, mode=mode)
    elif backbone == 'darknet53ss':
        alpha = 0.5
        # feature_size = 256
        backbone_m = cspdarknet53(image_input, alpha=0.5, mode=mode)
    else:
        assert 0

        # backbone_m.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    if freeze_backbone:
        for layer in backbone_m.layers:
            layer.trainable = False
    elif freeze_bn:
        for layer in backbone_m.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False

    outputs = backbone_m.outputs
    # []
    # FPN
    if fpn_type == 'dla':
        x = DLA_seg(outputs, down_ratio=4, mode=mode)
    elif fpn_type == 'simple':
        assert n_stage==3
        x = simple_fpn(outputs, int(feature_size*alpha))
    elif fpn_type == 'bifpn':
        x = bifpn(outputs, num_filters=128, mode=mode)
    elif fpn_type == 'pan':
        x = fpn_pan(outputs, alpha=alpha, feature_size=feature_size, mode=mode)
    else:
        x = fpn(outputs, alpha=alpha, feature_size=feature_size, mode=mode)
    features = x[:n_stage]
    print ("number of features", len(features))
    for item in features:
        print(item)

    # head
    regression = []
    classification = []
    centerness = []

    regression_debug = []
    classification_debug = []
    centerness_debug = []
    # regression
    for stage in range(len(features)):
        if head_type == 'simple':
            head1 = features[stage]
            head1 = head_models(head1, int(feature_size*alpha), stage="reghead_%d"%stage)
        else:
            head1 = head_model(features[stage], feature=256, stage="reghead_%d"%stage)
        reg = Conv2D(4, name='pyramid_regression_%d'%stage, **option_head)(head1)
        regression_debug.append(reg)
        # (b, num_anchors_this_feature_map, num_values)
        reg = Reshape((-1, 4), name='pyramid_regression_reshape_%d'%stage)(reg)
        # added for fcos
        if mapping_func == 'exp':
            reg = Lambda(lambda y: K.exp(y))(reg)
        else:
            reg = ReLU()(reg)
            reg = Lambda(lambda y: (2**(3+stage))*(y**2))(reg)
        regression.append(reg)

        # centerness
        if centerness_pos != 'cls':
            cts = Conv2D(1, name='pyramid_centerness_%d' % stage, **option_head)(head1)
            cts = Activation('sigmoid', name='pyramid_centerness_sigmoid_%d' % stage)(cts)
            centerness_debug.append(cts)
            # reshape output and apply sigmoid
            cts = Reshape((-1, 1), name='pyramid_centerness_reshape_%d' % stage)(cts)
            centerness.append(cts)
    
    # classification
    for stage in range(len(features)):
        if head_type == 'simple':
            head2 = features[stage]
            head2 = head_models(head2, int(feature_size*alpha), stage="clshead_%d" % stage)
        else:
            head2 = head_model(features[stage], feature=256, stage="clshead_%d"%stage)
        cls = Conv2D(num_classes, name='pyramid_classification_%d'%stage, **option_head)(head2)
        # reshape output and apply sigmoid
        cls = Activation('sigmoid', name='pyramid_classification_sigmoid_%d'%stage)(cls)
        classification_debug.append(cls)
        cls = Reshape((-1, num_classes), name='pyramid_classification_reshape_%d'%stage)(cls)
        classification.append(cls)

        #centerness
        if centerness_pos == 'cls':
            cts = Conv2D(1, name='pyramid_centerness_%d'%stage, **option_head)(head2)
            cts = Activation('sigmoid', name='pyramid_centerness_sigmoid_%d'%stage)(cts)
            centerness_debug.append(cts)
            # reshape output and apply sigmoid
            cts = Reshape((-1, 1), name='pyramid_centerness_reshape_%d'%stage)(cts)
            centerness.append(cts)
    reg = Concatenate(axis=1, name='regression')([f for f in regression])
    cls = Concatenate(axis=1, name='classification')([f for f in classification])
    cts = Concatenate(axis=1, name='centerness')([f for f in centerness])

    model = Model(inputs=[image_input],
                  outputs=[reg, cls, cts])
    debug_model = None
    if mode == 1:
        debug_model = Model(inputs=[image_input],
                            outputs=regression_debug+classification_debug+centerness_debug)
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=True)

    # prediction model
    print('training model output:')
    print(reg)
    print(cls)
    print(cts)

    import layers
    strides = (8, 16, 32, 64, 128)
    class_specific_filter = True
    locations = layers.Locations(strides=strides[:n_stage], name='locations')(features)
    # return keras.models.Model(inputs=model.inputs, outputs=locations, name=name)
    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([locations, reg])
    boxes = layers.ClipBoxes(name='clipped_boxes')([image_input, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = layers.FilterDetections(
        nms=nms,
        class_specific_filter=class_specific_filter,
        name='filtered_detections'
    )([boxes, cls, cts])
    prediction_model = Model(image_input, detections)

    return model, prediction_model, debug_model
