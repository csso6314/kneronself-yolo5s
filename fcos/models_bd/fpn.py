import keras
import tensorflow as tf
import layers
from keras.layers import *
from models_bd.utils import Act, get_bn_opt, get_conv_opt, get_convdw_opt

conv_option = get_conv_opt()
conv_dw_option = get_convdw_opt()
bn_option = get_bn_opt()


def simple_fpn(outputs,feature_size=128):
    from models_bd.darknet import conv, cspstage
    C3 = outputs[-3]#8
    C4 = outputs[-2]#16
    C5 = outputs[-1]#32
    P5 = cspstage(C5, feature_size, loop=1, mode=1, name='simplecsp_P5_')
    P4 = cspstage(C4, feature_size, loop=1, mode=1, name='simplecsp_P4_')
    P3 = cspstage(C3, feature_size, loop=1, mode=1, name='simplecsp_P3_')
    return [P3, P4, P5]

def fpn(outputs, alpha=1.0, feature_size=128,mode=0):
    feature_size = int(feature_size*alpha)
    C3 = outputs[-3]#8
    C4 = outputs[-2]#16
    C5 = outputs[-1]#32

    # upsample C5 to get P5 from the FPN paper
    P5 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    if mode == 0:
        P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    else:
        P5_upsampled = UpSampling2D(name='P5_upsampled')(P5)
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4 = Add(name='P4_merged')([P5_upsampled, P4])
    if mode == 0:
        P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    else:
        P4_upsampled = UpSampling2D(name='P4_upsampled')(P4)
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = Add(name='P3_merged')([P4_upsampled, P3])
    # P3_upsampled = UpSampling2D(name='P3_upsampled')(P3)
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(P5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    # x = Conv2D(feature_size*2, kernel_size=3, strides=1, padding='same', name='P3')(P3_upsampled)
    # x = BatchNormalization(**bn_option)(x)
    # x = ReLU()(x)
    return [P3, P4, P5, P6, P7]


def fpn_pan(outputs, alpha=1.0, feature_size=256, mode=0):
    from models_bd.darknet import conv, cspstage
    loops = 1
    # loops = [2,2]
    feature_size = int(alpha*feature_size)
    num_filters = [128*alpha, 256*alpha]
    num_filters = list(map(int, num_filters))

    C3 = outputs[-3]  # 8  128
    C4 = outputs[-2]  # 16 256
    C5 = outputs[-1]  # 32 256
    P_tds = [C5]

    # FPN
    # from C5 to C3
    # (16, 256)
    if mode == 0:
        P5_upsample = layers.UpsampleLike(name='fpn_P5_upsampling')([C5, C4])
    else:
        P5_upsample = UpSampling2D(name='fpn_P5_upsampling')(C5)
    P4 = Concatenate(axis=-1)([P5_upsample, C4]) # 16 512
    P4 = cspstage(P4, num_filters[-1], loop=loops, mode=1, name='fpn_cspstage_P4_') # 16 256
    P4 = conv(P4, num_filters[-1]//2, kernel=1, strides=1, padding=0, name='fpn_P4_conv') # 16 128
    P_tds.append(P4)

    if mode == 0:
        P4_upsample = layers.UpsampleLike(name='fpn_P4_upsampling')([P4, C3])
    else:
        P4_upsample = UpSampling2D(name='fpn_P4_upsampling')(P4) # 8 128
    P3 = Concatenate(axis=-1)([P4_upsample, C3]) # 8 256
    P3 = cspstage(P3, num_filters[-2], loop=loops, mode=1, name='fpn_cspstage_P3_') # 8 128
    P_tds.append(P3)
    # P3 = conv(P3, num_filters[-1] // 2, kernel=1, strides=1, padding=0, name='fpn_P4_conv')

    # PAN
    # from P3 to P5
    P_outputs = [P3]
    P5, P4, P3 = P_tds

    P3_down = conv(P3, num_filters[-2], kernel=3, strides=2, padding=1, name='pan_P3_down') # 16 128
    P4 = Concatenate(axis=-1)([P3_down, P4]) # 16 256
    P4 = cspstage(P4, num_filters[-1], loop=loops, mode=1, name='pan_cspstage_P4_') # 16 256
    P_outputs.append(P4)

    P4_down = conv(P4, num_filters[-1], kernel=3, strides=2, padding=1, name='pan_P4_down')# 32 256
    P5 = Concatenate(axis=-1)([P4_down, P5]) # 32 512
    P5 = cspstage(P5, num_filters[-1]*2, loop=loops, mode=1, name='pan_cspstage_P5_') # 32 512
    P_outputs.append(P5)

    #bottle neck
    P3, P4, P5 = P_outputs
    P3 = Conv2D(feature_size, kernel_size=1, strides=1, name='P3_output_%d' % feature_size)(P3)
    P4 = Conv2D(feature_size, kernel_size=1, strides=1, name='P4_output_%d' % feature_size)(P4)
    P5 = Conv2D(feature_size, kernel_size=1, strides=1, name='P5_output_%d' % feature_size)(P5)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(P_outputs[-1])

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)
    return [P3, P4, P5, P6, P7]


# bi-fpn
def _align_shape(inps, normalize=True):
    inps_n = []
    _, h1, w1, c1 = inps[0].shape.as_list()
    for inp in inps:
        _, h2, w2, c2 = inp.shape.as_list()
        assert c1==c2
        if h2 > h1 or w2 > w1:
            inp = Cropping2D(((0, h2-h1), (0, w2-w1)))(inp)
        else:
            assert h2 == h1 and w2 == w1
        if normalize:
            inp = BatchNormalization(**bn_option)(inp)
        inps_n.append(inp)
    return inps_n


def _bifpn_add(inps, num_filters, mode):
    conv_dw_option_2 =conv_dw_option.copy()
    conv_dw_option_2['use_bias'] =True
    assert(len(inps)>=2)
    if mode == 1:
        inps = _align_shape(inps)
    # should use weighted add
    bn_inps =[BatchNormalization(**bn_option)(item) for item in inps]
    x = Add()(bn_inps)
    # change from swish to prelu
    x = Act('prelu')(x)
    x = ZeroPadding2D()(x)
    x = DepthwiseConv2D(kernel_size=3, **conv_dw_option_2)(x)
    x = Conv2D(num_filters, kernel_size=1, **conv_option)(x)
    x = BatchNormalization(**bn_option)(x)
    return x


class _wBiFPNAdd(Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super(_wBiFPNAdd, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(shape=(num_in,),
                                 initializer=keras.initializers.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(_wBiFPNAdd, self).get_config()
        config.update({
            'epsilon': self.epsilon
        })


def bifpn(inputs, num_filters, block=None, mode=0):
    """
        P7_0 -------------------------- P7_2 -------->
        P6_0 ---------- P6_td ---------- P6_2 -------->
        P5_0 ---------- P5_1 ---------- P5_2 -------->
        P4_0 ---------- P4_1 ---------- P4_2 -------->
        P3_0 -------------------------- P3_2 -------->
    """
    # inputs should be P3 P4 P5 P6 P7
    # with largest stride P7
    assert len(inputs)>=2
    # get P_tds
    P_last = inputs[-1]
    _, h, w, c = P_last.shape.as_list()
    if c != num_filters:
        print('last output channel number do not match, expect %d, get %d'%(num_filters,c))
        P_last = Conv2D(num_filters, kernel_size=1, **conv_option)(P_last)
        P_last = BatchNormalization(**bn_option)(P_last)
        # act?
    P_tds = [P_last]

    for i in range(2, len(inputs)+1):
        # from P6(-2) to P3(-len)
        # get upsampling from next layer
        P_in = inputs[-i]
        if P_in.shape.as_list()[-1] != num_filters:
            P_in_sc = Conv2D(num_filters, kernel_size=1, **conv_option)(P_in)
            P_in_sc = BatchNormalization(**bn_option)(P_in_sc)
        else:
            P_in_sc = P_in
        if mode == 0:
            P_upsample = layers.UpsampleLike()([P_tds[-1], P_in_sc])
        else:
            P_upsample = UpSampling2D()(P_tds[-1])

        P_td = _bifpn_add([P_in_sc, P_upsample], num_filters, mode=mode)
        P_tds.append(P_td)
    # P3 to P7
    assert len(P_tds) == len(inputs)
    P_tds = P_tds[::-1]

    # get P_outputs
    P_outputs = [P_tds[0]]
    for i in range(1, len(inputs)-1):
        # from P3 to P7
        P_in = inputs[i]
        P_td = P_tds[i]
        if P_in.shape.as_list()[-1] != num_filters:
            P_in_sc = Conv2D(num_filters, kernel_size=1,**conv_option)(P_in)
            P_in_sc = BatchNormalization(**bn_option)(P_in_sc)
        else:
            P_in_sc = P_in
        P_downsample = MaxPool2D(pool_size=3,strides=2,padding='same')(P_outputs[-1])
        P_output = _bifpn_add([P_td, P_in_sc, P_downsample], num_filters)
        P_outputs.append(P_output)
    return P_outputs


if __name__ == '__main__':
    C3 = Input((76,76,128))
    C4 = Input((38,38,256))
    C5 = Input((19,19,256))
    C6 = Input((10, 10, 512))
    C7 = Input((5, 5, 512))
    # out = bifpn([C3,C4,C5,C6,C7], 64)
    # out = fpn_pan([C3,C4,C5],alpha=1.0,bottleneck=256,mode=1)
    out = fpn([C3,C4,C5],feature_size=256,mode=1)

    from keras.models import Model
    # m = Model([C3,C4,C5,C6,C7], out)
    m = Model([C3,C4,C5], out)
    m.summary()
    # m.save('fpn.hdf5')