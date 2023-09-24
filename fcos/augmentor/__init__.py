
from augmentor.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from augmentor.object_detection_2d_geometric_ops import Resize
from augmentor.object_detection_2d_patch_sampling_ops import RandomPadFixedAR
from augmentor.object_detection_2d_photometric_ops import ConvertTo3Channels

def val_aug(input_size=512):
    convert_to_3_channels = ConvertTo3Channels()
    pad = RandomPadFixedAR(1.0)
    resize = Resize(height=input_size, width=input_size)
    data_augmentation_chain2 = [convert_to_3_channels,
                                pad,
                                resize]
    return data_augmentation_chain2

def train_aug(input_size=512):
    data_augmentation_chain = DataAugmentationVariableInputSize(input_size,
                                                                input_size,
                                                                random_brightness=(-48, 48, 0.5),
                                                                random_contrast=(0.5, 1.8, 0.5),
                                                                random_saturation=(0.5, 1.8, 0.5),
                                                                random_hue=(18, 0.5),
                                                                random_flip=0.5,
                                                                n_trials_max=3,
                                                                min_scale=0.7,
                                                                max_scale=1.5,
                                                                min_aspect_ratio=0.8,
                                                                max_aspect_ratio=1.2,
                                                                clip_boxes=True,
                                                                overlap_criterion='area',
                                                                bounds_box_filter=(0.3, 1.0),
                                                                bounds_validator=(0.5, 1.0),
                                                                n_boxes_min=0,
                                                                background=(0, 0, 0))
    return  data_augmentation_chain.transformations


# def train_aug(input_size=512):
#     data_augmentation_chain = DataAugmentationVariableInputSize(input_size,
#                                                             input_size,
#                                                             random_brightness=(-128, 64, 0.5),
#                                                             random_contrast=(0.5, 1.8, 0.5),
#                                                             random_saturation=(0.5, 1.8, 0.5),
#                                                             random_hue=(18, 0.5),
#                                                             random_flip=0.5,
#                                                             n_trials_max=3,
#                                                             min_scale=0.1,
#                                                             max_scale=1.5,
#                                                             min_aspect_ratio = 0.7,
#                                                             max_aspect_ratio = 1.3,
#                                                             clip_boxes=True,
#                                                             overlap_criterion='area',
#                                                             bounds_box_filter=(0.3, 1.0),
#                                                             bounds_validator=(0.5, 1.0),
#                                                             n_boxes_min=0,
#                                                             background=(0,0,0))
    return  data_augmentation_chain.transformations