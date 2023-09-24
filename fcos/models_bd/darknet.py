import keras
from keras.layers import *
from keras.models import *
from models_bd.utils import Act, get_bn_opt, get_conv_opt, get_convdw_opt
import numpy as np

bn_option = get_bn_opt()
conv_option = get_conv_opt()
act_type = 'leaky'

def focus_block(data, mode=0):
    if mode == 0:
        x1 = Lambda(lambda x:x[:,::2,::2,:])(data)
        x2 = Lambda(lambda x:x[:,1::2,::2,:])(data)
        x3 = Lambda(lambda x:x[:,::2,1::2,:])(data)
        x4 = Lambda(lambda x:x[:,1::2,1::2,:])(data)
        x = Concatenate(axis=-1)([x1, x2, x3, x4])
    else:
        inp_dim = data.shape.as_list()[-1]
        focus_layer = Conv2D(12, kernel_size=3, strides=2, use_bias=False, name='focus_conv')
        x = ZeroPadding2D(((0,1),(0,1)))(data)
        x = focus_layer(x)
        weights = np.zeros((3, 3, inp_dim, 12))
        for i in range(4):
            for j in range(3):
                weights[i%2, i//2, j, j + i * 3] = 1
                weights[i%2, i//2, j, j + i * 3] = 1
                weights[i%2, i//2, j, j + i * 3] = 1
                weights[i%2, i//2, j, j + i * 3] = 1
        focus_layer.set_weights([weights])
        focus_layer.trainable = False
    return x

def conv(data, filter, kernel, strides, padding=0, use_bn=True, use_act=True, name=None):
    if padding>0:
        x = ZeroPadding2D(padding)(data)
    else:
        x = data
    x = Conv2D(filter, kernel, strides=strides, name=name, **conv_option)(x)
    if use_bn:
        x = BatchNormalization(name=name+'_bn', **bn_option)(x)
    if use_act:
        x = Act(act_type)(x)
    return x

def residual_block(data, num_filter, stride=1, use_sc=True, name=''):

    # conv-bn-relu-conv-bnjavascript:void(0)
    conv1 = data
    conv1 = conv(conv1, num_filter, kernel=1, strides=stride, padding=0, name=name + 'conv0')
    conv2 = conv(conv1, num_filter, kernel=3, strides=stride, padding=1, name=name + 'conv1')
    if use_sc:
        assert stride == 1
        out = Add()([conv2, data])
    else:
        out = conv2
    return out

def sppblock(data, filters, n_branch=4, name='spp_'):
    pool_sizes = [5, 9, 13]

    x = data
    x = conv(x, filters//2, kernel=1, strides=1, padding=0, name=name+'conv0')
    x_list = [x]
    if n_branch >= 2:
        x1 = ZeroPadding2D(2)(x)
        x1 = MaxPool2D(pool_sizes[0], strides=1, padding='valid', name=name+'pool1')(x1)
        x_list.append(x1)
    if n_branch >= 3:
        x2 = ZeroPadding2D(4)(x)
        x2 = MaxPool2D(pool_sizes[1], strides=1, padding='valid', name=name+'pool2')(x2)
        x_list.append(x2)
    if n_branch >= 4:
        x3 = ZeroPadding2D(6)(x)
        x3 = MaxPool2D(pool_sizes[2], strides=1, padding='valid', name=name+'pool3')(x3)
        x_list.append(x3)
    x = Concatenate(axis=-1)(x_list)
    x = conv(x, filters, kernel=1, strides=1, padding=0, name=name+'conv1')
    return x

def cspstage(data, filters, loop, mode=0, name=''):
    '''CSPNets stage
        param input_data: The input tensor
        param filters: Filter nums
        param loop: ResBlock loop nums
    return: Output tensors and the last Conv layer counter of this stage'''
    c = filters

    x = data
    x = conv(x, c//2, kernel=1, strides=1, padding=0, name=name+'conv0')
    for i in range(loop):
        if mode == 0:
            x = residual_block(x, c//2, name=name + 'residual%d_'%i)
        else:
            x = residual_block(x, c//2, use_sc=False, name=name + 'residual%d_'%i)
    x = conv(x, c//2, kernel=1, strides=1, padding=0,  use_bn=False, use_act=False, name=name+'conv1')

    # sc
    sc = conv(data, c // 2, kernel=1, strides=1, padding=0, use_bn=False, use_act=False, name=name + 'sc')

    # Concatenate
    x = Concatenate(axis=-1)([sc, x])
    x = BatchNormalization(name=name+'bn0', **bn_option)(x)
    x = Act(act_type)(x)
    x = conv(x, c, kernel=1, strides=1, padding=0, name=name+'conv2')
    return x


def cspdarknet53(shape, alpha=1.0, weights=None, include_top=True, n_class=1000, mode=0, **kwargs):
    channel = [max(32, 32*alpha), max(48, 64*alpha),128*alpha,256*alpha, 512*alpha]
    # channel = [32*alpha, 64*alpha,128*alpha,256*alpha, 512*alpha]

    channel = list(map(int, channel))
    if alpha == 1.0:
        loops = [1,3,3,1]
    elif alpha == 0.5:
        loops = [1, 2, 2, 1]
    else:
        loops = [2,6,6,2]
    outputs = []
    if isinstance(shape, tuple) or isinstance(shape, list):
        inp = Input(shape)
    else:
        assert keras.backend.is_keras_tensor(shape)
        inp = shape

    # 3x608x608 -> 12x304x304
    x = focus_block(inp, mode=mode)
    # 3x304x304 -> 32x304x304
    x = conv(x, channel[0], kernel=3, strides=1, padding=1, name='conv0')

    # 32x304x304 -> 64x152x152
    x = conv(x, channel[1], kernel=3, strides=2, padding=1, name='conv1')
    x = cspstage(x, channel[1], loop=loops[0], name='cspstage1_')
    outputs.append(x)

    # 64x152x152 -> 128x76x76
    x = conv(x, channel[2], kernel=3, strides=2, padding=1, name='conv2')
    x = cspstage(x, channel[2], loop=loops[1], name='cspstage2_')
    outputs.append(x)

    # 128x76x76 -> 256x38x38
    x = conv(x, channel[3], kernel=3, strides=2, padding=1, name='conv3')
    x = cspstage(x, channel[3], loop=loops[2], name='cspstage3_')
    outputs.append(x)

    # 256x38x38 -> 512x19x19
    x = conv(x, channel[4], kernel=3, strides=2, padding=1, name='conv4')
    x = sppblock(x, channel[4])
    x = cspstage(x, channel[4], loop=loops[3], mode=1, name='cspstage4_')
    x = conv(x, channel[4]//2, kernel=1, strides=1, padding=0, name='conv4_1')
    outputs.append(x)

    # output [C2, C3, C4, C5]
    m = Model(inp, outputs)
    if weights is not None:
        m.load_weights(weights)
    return m
if __name__ == '__main__':
    x = Input((320,320,3))
    m = cspdarknet53(x, alpha=0.5)
    m.summary()
    x = np.random.random((1,320,320,3))
    m.predict(x)

    import datetime
    t1_1 = datetime.datetime.now()
    for _ in range(100):
        m.predict(x)
    t2_1 = datetime.datetime.now()
    print("Single GPU computation time: " + str(t2_1 - t1_1))

    # m.save('./darknet53bb.hdf5')

    # x = Input((320,320,3))
    # x1 = focus_block(x,0)
    # x2 = focus_block(x,1)
    # m1 = Model(x,x1)
    # m2 = Model(x,x2)
    # a = np.random.rand(1,320,320,3)
    # assert np.all(m1.predict(a)==m2.predict(a))