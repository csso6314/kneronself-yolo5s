import numpy as np
from keras import layers
from keras.models import Model
from keras.layers import *
from keras.constraints import non_neg
from keras.regularizers import l2

try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations


def Act(act_type, name):
    if act_type == 'prelu':
        body = layers.PReLU(name=name, shared_axes=[1, 2], alpha_regularizer=l2(0.00004))
    elif act_type == 'leaky':
        body = layers.LeakyReLU(0.1171875, name=name + 'l')
    else:
        body = Activation(act_type, name=name)
    return body


def residual_unit_v3(data, num_filter, stride, dim_match, stage, block, **kwargs):
    use_se = kwargs.get('version_se', 0)
    bn_mom = kwargs.get('bn_mom', 0.9)
    act_type = kwargs.get('version_act', 'leaky')
    name_base = 'stage%d_block%d'%(stage, block)
    wd = kwargs.get('wd', 0.00004)
    epsilon = kwargs.get('epsilon', 2e-5)

    # bn-conv-bn-relu-conv-bn
    bn1 = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + '_bn1')(data)
    conv1 = Conv2D(filters=num_filter, kernel_size=3, strides=1, padding="same", use_bias=False, kernel_regularizer=l2(wd),
                   name=name_base + '_conv1')(bn1)
    bn2 = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + '_bn2')(conv1)
    act1 = Act(act_type=act_type, name=name_base + '_relu1')(bn2)
    act1 = layers.ZeroPadding2D(((1, 1), (1, 1)))(act1)
    conv2 = Conv2D(filters=num_filter, kernel_size=3, strides=stride, use_bias=False, kernel_regularizer=l2(wd),
                   name=name_base + '_conv2')(act1)
    bn3 = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + '_bn3')(conv2)

    if use_se:
        body = GlobalAveragePooling2D()(bn3)
        body = Dense(num_filter//8)(body)
        body = body = PReLU(alpha_regularizer=l2(0.00004), name=name_base+'se_relu1')(body)
        body = Dense(num_filter)(body)
        body = Activation('sigmoid')(body)
        bn3 = Multiply()([bn3, body])
    # short cut
    if dim_match:
        shortcut = data
    else:
        conv1sc = Conv2D(filters=num_filter, kernel_size=(1, 1), strides=stride, padding="valid", use_bias=False,
                         name=name_base + '_conv1sc')(data)
        shortcut = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + '_bnsc')(conv1sc)
    return layers.Add()([bn3, shortcut])


def base_model(input_img,  **kwargs):
    num_layers = kwargs.get('num_layers', 28)
    act_type = kwargs.get('version_act', 'leaky')
    bn_mom = kwargs.get('bn_mom', 0.9)
    wd = kwargs.get('wd', 0.00004)
    ft_mult = kwargs.get('ft_mult', 1.)
    epsilon = kwargs.get('epsilon', 2e-5)
    
    num_stages = 4
    if num_layers <= 101:
        filter_list = [32*ft_mult, 32*ft_mult, 64*ft_mult, 128*ft_mult, 256*ft_mult]
    else:
        filter_list = [64*ft_mult, 256*ft_mult, 512*ft_mult, 1024*ft_mult, 2048*ft_mult]
        bottle_neck = True
    filter_list = list(map(int, filter_list))
    if num_layers == 18:
        blocks = [2, 2, 2, 2]
    elif num_layers == 28:
        blocks = [3, 4, 3, 3]
    elif num_layers == 34:
        blocks = [3, 4, 6, 3]
    elif num_layers == 50:
        blocks = [3, 4, 14, 3]
    elif num_layers == 100:
        blocks = [3, 13, 30, 3]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
    residual_unit = residual_unit_v3
    assert (len(blocks) == num_stages)

    # building network
    #input_img = Input(input_shape)
    body = input_img

    body = Conv2D(filter_list[0], 7, strides=2, padding="same", use_bias=False, kernel_regularizer=l2(wd),
                  name='conv0')(body)
    body = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name='bn0')(body)
    body = Act(act_type=act_type, name='relu0')(body)
    # body = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(body)

    outputs = []
    for i in range(num_stages):
        body = residual_unit(body, filter_list[i + 1], 2, False, stage=i + 1, block=1, **kwargs)
        for j in range(blocks[i] - 1):
            body = residual_unit(body, filter_list[i + 1], 1, True, stage=i + 1, block=j + 2, **kwargs)
        outputs.append(body)
    model = Model(input_img, outputs)

    return model

if __name__ == '__main__':
    model = base_model((512,512,3))
    model.summary()
    print (model.outputs)
