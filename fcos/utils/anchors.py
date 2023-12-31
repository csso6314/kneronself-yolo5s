# -*- coding: utf-8 -*- 
import numpy as np

import keras

from utils.compute_overlap import compute_overlap


# class AnchorParameters:
#     """
#     The parameters that define how anchors are generated.
#
#     Args
#         sizes : List of sizes to use. Each size corresponds to one feature level.
#         strides : List of strides to use. Each stride correspond to one feature level.
#         ratios : List of ratios to use per location in a feature map.
#         scales : List of scales to use per location in a feature map.
#     """
#
#     def __init__(self, sizes, strides, ratios, scales, interest_sizes):
#         self.sizes = sizes
#         self.strides = strides
#         self.ratios = ratios
#         self.scales = scales
#         self.interest_sizes = interest_sizes
#
#     def num_anchors(self):
#         return len(self.ratios) * len(self.scales)

class AnchorParameters:
    """
    The parameters that define how anchors are generated.

    Args
        strides : List of strides to use. Each stride correspond to one feature level.
        scales : List of scales to use per location in a feature map.
    """

    def __init__(self, strides, interest_sizes):
        self.strides = strides
        self.interest_sizes = interest_sizes

"""
The default anchor parameters.
"""
AnchorParameters.default = AnchorParameters(
    # sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    # ratios=np.array([0.5, 1, 2], keras.backend.floatx()),
    # scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
    interest_sizes=[
        [-1, 64],
        [64, 128],
        [128, 256],
        [256, 512],
        [512, 1e8],
    ],
)


def anchor_targets_bbox(
        anchors,
        image_group,
        annotations_group,
        num_classes,
        negative_overlap=0.4,
        positive_overlap=0.5
):
    """
    Generate anchor targets for bbox detection.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        image_group: List of BGR images.
        annotations_group: List of annotations (np.array of shape (N, 5) for (x1, y1, x2, y2, label)).
        num_classes: Number of classes to predict.
        mask_shape: If the image is padded with zeros, mask_shape can be used to mark the relevant part of the image.
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        labels_batch: batch that contains labels & anchor states (np.array of shape (batch_size, N, num_classes + 1),
                      where N is the number of anchors for an image and the last column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg).
        regression_batch: batch that contains bounding-box regression targets for an image & anchor states (np.array of shape (batch_size, N, 4 + 1),
                      where N is the number of anchors for an image, the first 4 columns define regression targets for (x1, y1, x2, y2) and the
                      last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """

    assert (len(image_group) == len(annotations_group)), "The length of the images and annotations need to be equal."
    assert (len(annotations_group) > 0), "No data received to compute anchor targets for."
    for annotations in annotations_group:
        assert ('bboxes' in annotations), "Annotations should contain bboxes."
        assert ('labels' in annotations), "Annotations should contain labels."

    batch_size = len(image_group)

    regression_batch = np.zeros((batch_size, anchors.shape[0], 4 + 1), dtype=keras.backend.floatx())
    labels_batch = np.zeros((batch_size, anchors.shape[0], num_classes + 1), dtype=keras.backend.floatx())

    # compute labels and regression targets
    for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
        if annotations['bboxes'].shape[0]:
            # obtain indices of gt annotations with the greatest overlap
            positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(anchors,
                                                                                            annotations['bboxes'],
                                                                                            negative_overlap,
                                                                                            positive_overlap)
            labels_batch[index, ignore_indices, -1] = -1
            labels_batch[index, positive_indices, -1] = 1

            regression_batch[index, ignore_indices, -1] = -1
            regression_batch[index, positive_indices, -1] = 1

            # compute target class labels
            labels_batch[
                index, positive_indices, annotations['labels'][argmax_overlaps_inds[positive_indices]].astype(int)] = 1

            regression_batch[index, :, :-1] = bbox_transform(anchors, annotations['bboxes'][argmax_overlaps_inds, :])

        # ignore anchors outside of image
        if image.shape:
            anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
            indices = np.logical_or(anchors_centers[:, 0] >= image.shape[1], anchors_centers[:, 1] >= image.shape[0])

            labels_batch[index, indices, -1] = -1
            regression_batch[index, indices, -1] = -1

    return regression_batch, labels_batch


def compute_gt_annotations(
        anchors,
        annotations,
        negative_overlap=0.4,
        positive_overlap=0.5
):
    """
    Obtain indices of gt annotations with the greatest overlap.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        annotations: np.array of shape (K, 5) for (x1, y1, x2, y2, label).
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        positive_indices: indices of positive anchors
        ignore_indices: indices of ignored anchors
        argmax_overlaps_inds: ordered overlaps indices
    """

    overlaps = compute_overlap(anchors.astype(np.float64), annotations.astype(np.float64))
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    # assign "dont care" labels
    positive_indices = max_overlaps >= positive_overlap
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, argmax_overlaps_inds


def layer_shapes(image_shape, model):
    """
    Compute layer shapes given input image shape and the model.

    Args
        image_shape: The shape of the image.
        model: The model to use for computing how the image shape is transformed in the pyramid.

    Returns
        A dictionary mapping layer names to image shapes.
    """
    shape = {
        model.layers[0].name: (None,) + image_shape,
    }

    for layer in model.layers[1:]:
        nodes = layer._inbound_nodes
        for node in nodes:
            input_shapes = [shape[inbound_layer.name] for inbound_layer in node.inbound_layers]
            if not input_shapes:
                continue
            shape[layer.name] = layer.compute_output_shape(input_shapes[0] if len(input_shapes) == 1 else input_shapes)

    return shape


def make_shapes_callback(model):
    """
    Make a function for getting the shape of the pyramid levels.
    """

    def get_shapes(image_shape, pyramid_levels):
        shape = layer_shapes(image_shape, model)
        image_shapes = [shape["P{}".format(level)][1:3] for level in pyramid_levels]
        return image_shapes

    return get_shapes


def guess_shapes(image_shape, pyramid_levels=(3, 4, 5, 6, 7)):
    """
    Guess shapes based on pyramid levels.

    Args
         image_shape: The shape of the image.
         pyramid_levels: A list of what pyramid levels are used.

    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    feature_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return feature_shapes


def compute_locations_per_level(h, w, stride):
    # [0, 8, 16]
    shifts_x = np.arange(0, w * stride, step=stride, dtype=np.float32)
    # [0, 8, 16, 24]
    shifts_y = np.arange(0, h * stride, step=stride, dtype=np.float32)
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    # (h * w, )
    shift_x = shift_x.reshape(-1)
    # (h * w, )
    shift_y = shift_y.reshape(-1)
    locations = np.stack((shift_x, shift_y), axis=1) + stride // 2
    return locations


def compute_locations(feature_shapes, anchor_param):
    """

    Args:
        feature_shapes: list of (h, w)
        anchor_params: instance of AnchorParameters

    Returns:
        locations: list of np.array (shape is (fh * fw, 2))

    """
    if anchor_param is None:
        anchor_param = AnchorParameters.default
    fpn_strides = anchor_param.strides
    n_stage = len(fpn_strides)
    # print('fpn stides in generator', fpn_strides)
    locations = []
    for level, (feature_shape, fpn_stride) in enumerate(zip(feature_shapes[:n_stage], fpn_strides)):
        h, w = feature_shape
        locations_per_level = compute_locations_per_level(
            h, w, fpn_stride
        )
        locations.append(locations_per_level)
    return locations


def compute_interest_sizes(num_locations_each_level, anchor_param):
    """

    Args:
        num_locations_each_level: list of int
        anchor_param:

    Returns:
        interest_sizes (np.array): (sum(fh * fw), 2)

    """
    if anchor_param is None:
        anchor_param = AnchorParameters.default
    interest_sizes = anchor_param.interest_sizes
    # print('interest_sizes in generator', interest_sizes)

    assert len(num_locations_each_level) == len(interest_sizes)
    tiled_interest_sizes = []
    for num_locations, interest_size in zip(num_locations_each_level, interest_sizes):
        interest_size = np.array(interest_size)
        interest_size = np.expand_dims(interest_size, axis=0)
        interest_size = np.tile(interest_size, (num_locations, 1))
        tiled_interest_sizes.append(interest_size)
    interest_sizes = np.concatenate(tiled_interest_sizes, axis=0)
    return interest_sizes

def get_sample_region(gt, anchor_param, num_points_per, cx, cy, radius=1.5):
    '''
    gt: (n, 4)
    strides: []
    num_points_per: []
    gt_xs: (m,)
    gt_ys: (m,)
    This code is from
    https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
    maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
    '''
    if anchor_param is None:
        anchor_param = AnchorParameters.default
    strides = anchor_param.strides
    n = gt.shape[0]
    # num of position
    m = len(cx)
    # (m, n, 4)
    gt = np.tile(gt[None], (m,1,1))
    assert gt.shape==(m,n,4)
    # (m, n)
    center_x = (gt[..., 0] + gt[..., 2]) / 2
    # (m, n)
    center_y = (gt[..., 1] + gt[..., 3]) / 2
    # (m, n, 4)
    center_gt = np.zeros(gt.shape)
    # no gt
    if center_x[..., 0].sum() == 0:
        return np.zeros((m, n), dtype=np.uint8)
    beg = 0
    for level, n_p in enumerate(num_points_per):
        end = beg + n_p
        stride = strides[level] * radius
        xmin = center_x[beg:end] - stride
        ymin = center_y[beg:end] - stride
        xmax = center_x[beg:end] + stride
        ymax = center_y[beg:end] + stride
        # limit sample region in gt
        center_gt[beg:end, :, 0] = np.where(
            xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
        )
        center_gt[beg:end, :, 1] = np.where(
            ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
        )
        center_gt[beg:end, :, 2] = np.where(
            xmax > gt[beg:end, :, 2],
            gt[beg:end, :, 2], xmax
        )
        center_gt[beg:end, :, 3] = np.where(
            ymax > gt[beg:end, :, 3],
            gt[beg:end, :, 3], ymax
        )
        beg = end
    # (m, n) - (1, n) --> (m, n)
    left = cx[:, None] - center_gt[..., 0]
    top = cy[:, None] - center_gt[..., 1]
    # (m, n) - (m, 1) --> (m, n)
    right = center_gt[..., 2] - cx[:, None]
    bottom = center_gt[..., 3] - cy[:, None]
    # (m,n,4)
    center_bbox = np.stack((left, top, right, bottom), -1)
    inside_gt_bbox_mask = center_bbox.min(axis=2) > 0
    return inside_gt_bbox_mask

def anchors_for_shape(
        image_shape,
        pyramid_levels=None,
        anchor_params=None,
        shapes_callback=None,
):
    """
    Generators anchors for a given shape.

    Args
        image_shape: The shape of the image.
        pyramid_levels: List of ints representing which pyramids to use (defaults to [3, 4, 5, 6, 7]).
        anchor_params: Struct containing anchor parameters. If None, default values are used.
        shapes_callback: Function to call for getting the shape of the image at different pyramid levels.

    Returns
        np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates for the anchors.
    """

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if shapes_callback is None:
        shapes_callback = guess_shapes
    feature_map_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales
        )
        shifted_anchors = shift(feature_map_shapes[idx], anchor_params.strides[idx], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(feature_map_shape, stride, anchors):
    """
    Produce shifted anchors based on shape of the map and stride size.

    Args
        feature_map_shape : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, feature_map_shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, feature_map_shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X scales w.r.t. a reference window.

    Args:
        base_size:
        ratios:
        scales:

    Returns:
        anchors: (num_anchors, 4), 4 为以 (0, 0) 为中心点的矩形坐标 (-w/2, -h/2, w/2, h/2)

    """
    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    # (num_anchors, )
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    # (num_anchors, )
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (cx, cy, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    """
    Args:
        anchors: (N, 4)
        gt_boxes: (N, 4)
        mean:
        std:

    Returns:

    """

    if mean is None:
        mean = np.array([0, 0, 0, 0])
    if std is None:
        std = np.array([0.2, 0.2, 0.2, 0.2])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights
    # (4, N)
    targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2))
    # (N, 4)
    targets = targets.T

    targets = (targets - mean) / std

    return targets
