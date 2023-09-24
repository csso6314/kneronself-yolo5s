import tensorflow as tf
from keras.layers import *
from keras.regularizers import *


bn_mom = 0.99
wd = 0.0001
epsilon = 2e-5
def get_conv_opt(bias=False):
    conv_option = {'kernel_regularizer': l2(wd),
                   'kernel_initializer': 'he_normal',
                   'use_bias': bias}
    return conv_option

def get_convdw_opt():
    conv_dw_option = {'depthwise_regularizer': l2(wd),
                      'kernel_initializer': 'he_normal',
                      'use_bias': False}
    return conv_dw_option

def get_bn_opt():
    bn_option = {'epsilon': epsilon,
                 'momentum': bn_mom}
    return bn_option

def Act(act_type):
    if act_type == 'prelu':
        body = PReLU(shared_axes=[1, 2], alpha_regularizer=l2(0.00004))
    elif act_type == 'leaky':
        body = LeakyReLU(0.1171875)
    else:
        body = Activation(act_type)
    return body

def nms(heat, kernel=3):
    hmax = MaxPool2D(kernel, strides=1, padding='same')(heat)
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat


def topk(hm, max_objects=100):
    hm = nms(hm)
    # (b, h * w * c)
    b, h, w, c = hm.shape.as_list()
    # hm2 = tf.transpose(hm, (0, 3, 1, 2))
    # hm2 = tf.reshape(hm2, (b, c, -1))
    hm = Reshape((-1,))(hm)
    # (b, k), (b, k)
    scores, indices = tf.nn.top_k(hm, k=max_objects)
    # scores2, indices2 = tf.nn.top_k(hm2, k=max_objects)
    # scores2 = tf.reshape(scores2, (b, -1))
    # topk = tf.nn.top_k(scores2, k=max_objects)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys


def evaluate_batch_item(batch_item_detections, num_classes, max_objects_per_class=20, max_objects=100,
                        iou_threshold=0.5, score_threshold=0.1):
    batch_item_detections = tf.boolean_mask(batch_item_detections,
                                            tf.greater(batch_item_detections[:, 4], score_threshold))
    detections_per_class = []
    for cls_id in range(num_classes):
        class_detections = tf.boolean_mask(batch_item_detections, tf.equal(batch_item_detections[:, 5], cls_id))
        nms_keep_indices = tf.image.non_max_suppression(class_detections[:, :4],
                                                        class_detections[:, 4],
                                                        max_objects_per_class,
                                                        iou_threshold=iou_threshold)
        class_detections = K.gather(class_detections, nms_keep_indices)
        detections_per_class.append(class_detections)

    batch_item_detections = K.concatenate(detections_per_class, axis=0)

    def filter():
        # nonlocal batch_item_detections
        _, indices = tf.nn.top_k(batch_item_detections[:, 4], k=max_objects)
        batch_item_detections_ = tf.gather(batch_item_detections, indices)
        return batch_item_detections_

    def pad():
        # nonlocal batch_item_detections
        batch_item_num_detections = tf.shape(batch_item_detections)[0]
        batch_item_num_pad = tf.maximum(max_objects - batch_item_num_detections, 0)
        batch_item_detections_ = tf.pad(tensor=batch_item_detections,
                                        paddings=[
                                            [0, batch_item_num_pad],
                                            [0, 0]],
                                        mode='CONSTANT',
                                        constant_values=0.0)
        return batch_item_detections_

    batch_item_detections = tf.cond(tf.shape(batch_item_detections)[0] >= 100,
                                    filter,
                                    pad)
    return batch_item_detections


def decode(hm, wh, reg, max_objects=100, nms=True, flip_test=False, num_classes=20, score_threshold=0.1,
           cat_spec_wh=True):
    if flip_test:
        hm = (hm[0:1] + hm[1:2, :, ::-1]) / 2
        wh = (wh[0:1] + wh[1:2, :, ::-1]) / 2
        reg = reg[0:1]
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b, h, w, c = hm._keras_shape
    # (b, h * w, 2)
    reg = Reshape((h * w, reg._keras_shape[-1]))(reg)
    # (b, h * w, 2*c) or (b, h * w, 2)
    wh = Reshape((h * w, wh._keras_shape[-1]))(wh)
    # (b, k, 2)
    ii = tf.tile(tf.range(tf.shape(reg)[0])[:, tf.newaxis], (1, max_objects))
    idx = tf.stack([ii, indices], axis=-1)

    # (b, k, 2)
    topk_reg = tf.gather_nd(reg, idx)
    topk_wh = tf.cast(tf.gather_nd(wh, idx), tf.float32)
    if not cat_spec_wh:
        topk_wh = Reshape((max_objects, 2))(topk_wh)
    else:
        # (b, k, c, 2)
        ii = tf.tile(tf.range(tf.shape(reg)[0])[:, tf.newaxis], (1, max_objects))
        jj = tf.tile(tf.range(max_objects)[tf.newaxis, :], (tf.shape(reg)[0], 1))
        idx = tf.stack([ii, jj, class_ids], axis=-1)
        topk_wh = Reshape((max_objects, c, 2))(topk_wh)
        topk_wh = tf.cast(tf.gather_nd(topk_wh, idx), tf.float32)
        topk_wh = Reshape((max_objects, 2))(topk_wh)

    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    topk_x1 = topk_cx - topk_wh[..., 0:1] / 2
    topk_x2 = topk_cx + topk_wh[..., 0:1] / 2
    topk_y1 = topk_cy - topk_wh[..., 1:2] / 2
    topk_y2 = topk_cy + topk_wh[..., 1:2] / 2
    # (b, k, 6)
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)
    if nms:
        detections = tf.map_fn(lambda x: evaluate_batch_item(x[0],
                                                             num_classes=num_classes,
                                                             score_threshold=score_threshold),
                               elems=[detections],
                               dtype=tf.float32)
    return detections

def decode_ltrb(hm, reg, max_objects=100, nms=True, flip_test=False, num_classes=20, score_threshold=0.1,
           cat_spec_wh=True):
    if flip_test:
        hm = (hm[0:1] + hm[1:2, :, ::-1]) / 2
        reg = reg[0:1]
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    ii = tf.tile(tf.range(tf.shape(reg)[0])[:, tf.newaxis], (1, max_objects))
    # (b, k, 3)
    idx = tf.stack([ii, ys, xs], axis=-1)
    # (b, k, 2)
    topk_reg = tf.gather_nd(reg, idx)
    print(topk_reg.shape)
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    topk_x1 = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) - topk_reg[..., 0:1]
    topk_x2 = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 2:3]
    topk_y1 = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) - topk_reg[..., 1:2]
    topk_y2 = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 3:4]
    # (b, k, 6)
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)
    if nms:
        detections = tf.map_fn(lambda x: evaluate_batch_item(x[0],
                                                             num_classes=num_classes,
                                                             score_threshold=score_threshold),
                               elems=[detections],
                               dtype=tf.float32)
    return detections