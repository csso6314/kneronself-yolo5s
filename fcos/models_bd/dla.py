from keras.layers import *
from keras.models import *
from keras.regularizers import *

import numpy as np
import keras
from models_bd.utils import Act, get_bn_opt, get_conv_opt, get_convdw_opt

conv_option = get_conv_opt()
conv_dw_option = get_convdw_opt()
bn_option = get_bn_opt()
act_type = 'relu'

def residual_block(data, num_filter,stride=1, dialation=1, residual=None):

    # conv-bn-relu-conv-bnjavascript:void(0)
    conv1 = data
    conv1 = ZeroPadding2D(1)(conv1)
    conv1 = Conv2D(filters=num_filter, kernel_size=3, strides=stride, **conv_option)(conv1)
    bn2 = BatchNormalization(**bn_option)(conv1)
    act1 = Act(act_type=act_type)(bn2)

    act1 = ZeroPadding2D(1)(act1)
    conv2 = Conv2D(filters=num_filter, kernel_size=3, strides=1, dilation_rate=dialation, **conv_option)(act1)
    bn3 = BatchNormalization(**bn_option)(conv2)
    if residual is not None:
        out = Add()([bn3, residual])
    else:
        assert stride==1
        out = Add()([bn3, data])
    out = Act(act_type)(out)
    return out


def root(children, filters, kernel, residual):
    x = children
    x = Concatenate(axis=-1)(x)
    x = ZeroPadding2D((kernel-1)//2)(x)
    x = Conv2D(filters, kernel, **conv_option)(x)
    x = BatchNormalization(**bn_option)(x)
    if residual:
        x = Add()([x, children[0]])
    x = Act(act_type)(x)
    return x


def Tree(data, levels, block, num_filter, stride=1,  root_kernel_size=1, dilation=1,
         level_root=False, root_residual=False, children=None):

    children = [] if children is None else children
    in_channel = data._keras_shape[-1]
    if stride > 1:

        bottom = MaxPool2D(stride, padding='same')(data)
    else:
        bottom = data

    if in_channel != num_filter:
        residual = Conv2D(num_filter, kernel_size=1)(bottom)
        residual = BatchNormalization(**bn_option)(residual)
    else:
        residual = bottom

    if level_root:
        children.append(bottom)

    if levels == 1:
        x1 = block(data, num_filter, stride, dilation, residual)
        x2 = block(x1, num_filter,   1,      dilation)
        x = root([x2, x1]+children, num_filter, root_kernel_size, root_residual)
    else:
        x1 = Tree(data, levels-1, block, num_filter, stride=stride, root_kernel_size=root_kernel_size, dilation=dilation,
                  level_root=False, root_residual=root_residual, children=None)
        children.append(x1)
        x = Tree(x1, levels-1, block, num_filter, stride=1, root_kernel_size=root_kernel_size, dilation=dilation,
                 level_root=False, root_residual=root_residual, children=children)

    return x

def DLA(shape, n_layer=34, block=residual_block, root_residual=False, include_top=True, n_class=1000, **kwargs):

    if n_layer == 34:
        levels = [1, 1, 1, 2, 2, 1]
        channels =  [16, 32, 64, 128, 256, 512]
    if isinstance(shape,tuple) or isinstance(shape,list):
        inp = Input(shape)
    else:
        assert keras.backend.is_keras_tensor(shape)
        inp = shape
    conv1 = Conv2D(channels[0], kernel_size=7, strides=1, padding="same", **conv_option)(inp)
    bn1 = BatchNormalization(**bn_option)(conv1)
    act1 = Act(act_type)(bn1)

    conv2 = Conv2D(channels[0], kernel_size=3, strides=1, padding="same", **conv_option)(act1)
    bn2 = BatchNormalization(**bn_option)(conv2)
    act2 = Act(act_type)(bn2)
    level0 = act2

    conv3 = Conv2D(channels[1], kernel_size=3, strides=2, padding="same", **conv_option)(act2)
    bn3 = BatchNormalization(**bn_option)(conv3)
    act3 = Act(act_type)(bn3)
    level1 = act3

    level2 = Tree(level1, levels[2], block, channels[2], stride=2,  root_kernel_size=1, dilation=1,
                  level_root=False, root_residual=root_residual, children=None)
    level3 = Tree(level2, levels[3], block, channels[3], stride=2, root_kernel_size=1, dilation=1,
                  level_root=True, root_residual=root_residual, children=None)
    level4 = Tree(level3, levels[4], block, channels[4], stride=2, root_kernel_size=1, dilation=1,
                  level_root=True, root_residual=root_residual, children=None)
    level5 = Tree(level4, levels[5], block, channels[5], stride=2, root_kernel_size=1, dilation=1,
                  level_root=True, root_residual=root_residual, children=None)
    if not include_top:
        x = [level0, level1, level2, level3, level4, level5]
    else:
        x = GlobalAveragePooling2D()(level5)
        x = Dense(n_class)(x)

    m = Model(inp, x)
    return m

def IDAUp(layers, node_kernel, num_filter, up_factors):
    """
    aggregate all layers and make nodes
    first layer should have smaller stride
    :param layers:
    :param node_kernel:
    :param num_filter:
    :param up_factors:
    :return:
    """
    # every layer do a upsample and projection except first layer

    for i, l in enumerate(layers):
        in_channel = l._keras_shape[-1]
        x = l

        if in_channel != num_filter:
            x = Conv2D(num_filter, kernel_size=1, **conv_option)(x)
            x = BatchNormalization(**bn_option)(x)
            x = Act(act_type)(x)

        if up_factors[i] != 1:
            x = UpSampling2D()(x)
            x = ZeroPadding2D((up_factors[i]*2+1)//2)(x)
            x = DepthwiseConv2D(kernel_size=up_factors[i]*2+1)(x)

        # replace old layer with new one
        layers[i] = x
    # all layer should have same size(h,w,c)

    x = layers[0]
    y = []
    # then aggregate all layer and make nodes
    for i in range(1,len(layers)):
        # every step aggregate two layer and make a node x
        x = Concatenate(axis=-1)([x, layers[i]])
        x = ZeroPadding2D(node_kernel//2)(x)
        x = Conv2D(num_filter, kernel_size=node_kernel,**conv_option)(x)
        x = BatchNormalization(**bn_option)(x)
        x = Act(act_type)(x)
        y.append(x)
    return x, y


def DLA_up(layers, channels, scales=(1,2,4,8,16)):
    """
    deep aggregation
    scales update
    step1:
        8, 16 -> 8, 8
    step2:
        4, 8, 8 -> 4, 4, 4
    step3:
        2, 4, 4, 4 -> 2 2 2 2
    step4:
        1 2 2 2 2 -> 1 1 1 1 1
    :param layers: output from backbone model
    :param channels: output channel for each step
    :param scales: stride of each layers
    :return:
    """
    layers = list(layers)
    assert (len(layers)==len(scales) and len(layers)==len(channels))
    scales = np.asarray(scales, 'int')
    assert (len(layers) > 1)
    x = layers[-1]
    for i in range(len(layers) - 1):
        # get last several layer to aggregation from 2 to all
        j = -i-2
        x, y = IDAUp(layers[j:], node_kernel=3, num_filter=channels[j], up_factors=scales[j:]//scales[j])
        # update layers with new node update scale with new scale
        layers[j+1:] = y
        scales[j+1:] = scales[j]
    return x


def DLA_seg(layers, down_ratio=4,mode=0):

    assert (down_ratio in [2,4,8,16])
    first_level = int(np.log2(down_ratio))

    # print layers
    channels = [item._keras_shape[-1] for item in layers]
    scales = [2**i for i in range(len(channels[first_level:]))]

    x = DLA_up(layers[first_level:], channels[first_level:], scales)

    return x


if __name__ == '__main__':
    m = DLA((512,512,3))
    m.summary()

    x = np.random.random((1, 512, 512, 3))
    m.predict(x)

    import datetime
    t1_1 = datetime.datetime.now()
    for _ in range(100):
        m.predict(x)
    t2_1 = datetime.datetime.now()
    print("Single GPU computation time: " + str(t2_1 - t1_1))
    # model.save('dla34.hdf5')