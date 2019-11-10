import tensorflow as tf
import tensorlayer as tl
import numpy as np

Kc = 64

def model(x, training=True, tb=True, alpha=0.0):

  with tf.name_scope('colorization'):
    # Low-level features net
    network = conv2d(x,       [3, 3, 1, Kc],      'conv11', training, tb, True, alpha, 2)
    network = conv2d(network, [3, 3, Kc, 2*Kc],   'conv21', training, tb, True, alpha)
    network = conv2d(network, [3, 3, 2*Kc, 2*Kc], 'conv22', training, tb, True, alpha, 2)
    network = conv2d(network, [3, 3, 2*Kc, 4*Kc], 'conv31', training, tb, True, alpha)
    network = conv2d(network, [3, 3, 4*Kc, 4*Kc], 'conv32', training, tb, True, alpha, 2)
    network = conv2d(network, [3, 3, 4*Kc, 8*Kc], 'conv41', training, tb, True, alpha)

    # Mid-level features net
    network = conv2d(network, [3, 3, 8*Kc, 8*Kc], 'conv42', training, tb, True, alpha)
    network = conv2d(network, [3, 3, 8*Kc, 4*Kc], 'conv43', training, tb, True, alpha)

    _, sx, sy, _ = network.shape.as_list()

    # Colorization network
    network = conv2d(network, [3, 3, 4*Kc, 2*Kc], 'conv44', training, tb, True, alpha)
    network = tf.image.resize_images(network, size=[int(2*sx), int(2*sy)], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    network = conv2d(network, [3, 3, 2*Kc, Kc],   'conv51', training, tb, True, alpha)
    network = conv2d(network, [3, 3, Kc, Kc],     'conv52', training, tb, True, alpha)
    network = tf.image.resize_images(network, size=[int(4*sx), int(4*sy)], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    network = conv2d(network, [3, 3, Kc, int(Kc/2)],   'conv61', training, tb, True, alpha)
    network = conv2d(network, [3, 3, int(Kc/2), 2],     'conv62', training, tb, True, alpha)

  return network


def model_vgg(x, training=True, tb=True, alpha=0.0):

    # Encoder network (VGG16, until pool5)
    x_in = tf.scalar_mul(255.0, x)
    net_in = tl.layers.InputLayer(x_in, name='input_layer')
    conv_layers = encoder(net_in)

    # Fully convolutional layers on top of VGG16 conv layers
    network = tl.layers.Conv2dLayer(conv_layers,
                    act = tf.identity,
                    shape = [3, 3, 512, 512],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='encoder/h6/conv')
    network = tl.layers.BatchNormLayer(network, is_train=training, name='encoder/h6/batch_norm')
    network.outputs = tf.nn.relu(network.outputs, name='encoder/h6/relu')

    # Decoder network
    network = decoder(network.outputs, training, tb, alpha)

    if training:
        return network, conv_layers

    return network


def encoder(input_layer):

    VGG_MEAN = [103.939, 116.779, 123.68]

    # Convert RGB to BGR
    red, green, blue = tf.split(input_layer.outputs, 3, 3)
    bgr = tf.concat([ blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2] ], axis=3)

    network = tl.layers.InputLayer(bgr, name='encoder/input_layer_bgr')

    # Convolutional layers size 1
    network     = conv_layer(network, [ 3, 64], 'encoder/h1/conv_1')
    beforepool1 = conv_layer(network, [64, 64], 'encoder/h1/conv_2')
    network     = pool_layer(beforepool1, 'encoder/h1/pool')

    # Convolutional layers size 2
    network     = conv_layer(network, [64, 128], 'encoder/h2/conv_1')
    beforepool2 = conv_layer(network, [128, 128], 'encoder/h2/conv_2')
    network     = pool_layer(beforepool2, 'encoder/h2/pool')

    # Convolutional layers size 3
    network     = conv_layer(network, [128, 256], 'encoder/h3/conv_1')
    network     = conv_layer(network, [256, 256], 'encoder/h3/conv_2')
    beforepool3 = conv_layer(network, [256, 256], 'encoder/h3/conv_3')
    network     = pool_layer(beforepool3, 'encoder/h3/pool')

    # Convolutional layers size 4
    network     = conv_layer(network, [256, 512], 'encoder/h4/conv_1')
    network     = conv_layer(network, [512, 512], 'encoder/h4/conv_2')
    beforepool4 = conv_layer(network, [512, 512], 'encoder/h4/conv_3')
    network     = pool_layer(beforepool4, 'encoder/h4/pool')

    # Convolutional layers size 5
    network     = conv_layer(network, [512, 512], 'encoder/h5/conv_1')
    network     = conv_layer(network, [512, 512], 'encoder/h5/conv_2')
    beforepool5 = conv_layer(network, [512, 512], 'encoder/h5/conv_3')
    network     = pool_layer(beforepool5, 'encoder/h5/pool')

    return network


def decoder(network, training=True, tb=True, alpha=0.0):

  with tf.name_scope('colorization'):
    _, sx, sy, _ = network.shape.as_list()

    # Colorization network
    network = tf.image.resize_images(network, size=[int(2*sx), int(2*sy)], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    network = conv2d(network, [3, 3, 8*Kc, 4*Kc], 'conv51', training, tb, True, alpha)
    network = conv2d(network, [3, 3, 4*Kc, 4*Kc], 'conv52', training, tb, True, alpha)
    network = tf.image.resize_images(network, size=[int(4*sx), int(4*sy)], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    network = conv2d(network, [3, 3, 4*Kc, 2*Kc], 'conv61', training, tb, True, alpha)
    network = conv2d(network, [3, 3, 2*Kc, 2*Kc], 'conv62', training, tb, True, alpha)
    network = tf.image.resize_images(network, size=[int(8*sx), int(8*sy)], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    network = conv2d(network, [3, 3, 2*Kc, Kc], 'conv71', training, tb, True, alpha)
    network = conv2d(network, [3, 3, Kc, Kc], 'conv72', training, tb, True, alpha)
    network = tf.image.resize_images(network, size=[int(16*sx), int(16*sy)], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    network = conv2d(network, [3, 3, Kc, int(Kc/2)],   'conv81', training, tb, True, alpha)
    network = conv2d(network, [3, 3, int(Kc/2), 2],     'conv82', training, tb, True, alpha)

  return network


# Load weights for VGG16 encoder convolutional layers
# Weights are from a .npy file generated with the caffe-tensorflow tool
def load_vgg_weights(network, weight_file, session):
    params = []

    if weight_file.lower().endswith('.npy'):
        npy = np.load(weight_file, encoding='latin1')
        for key, val in sorted(npy.item().items()):
            if(key[:4] == "conv"):
                print("  Loading %s" % (key))
                print("  weights with size %s " % str(val['weights'].shape))
                print("  and biases with size %s " % str(val['biases'].shape))
                params.append(val['weights'])
                params.append(val['biases'])
    else:
        print('No weights in suitable .npy format found for path ', weight_file)

    print('Assigning loaded weights..')
    tl.files.assign_params(session, params, network)

    return network


# Count the number of weights in the NN
def count_all_vars():
  N = 0
  for v in tf.global_variables():
    N += np.prod(v.shape.as_list())

  return N


# === Layers ==================================================================

# TL convolutional layer
def conv_layer(input_layer, sz, str):
    network = tl.layers.Conv2dLayer(input_layer,
                    act = tf.nn.relu,
                    shape = [3, 3, sz[0], sz[1]],
                    strides = [1, 1, 1, 1],
                    padding = 'SAME',
                    name = str)

    return network

# Max-pooling layer
def pool_layer(input_layer, str):
    network = tl.layers.PoolLayer(input_layer,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name = str)

    return network

def conv2d(x, s, nmn, training=True, tb=True, bn=True, alpha=0.0, strd=1):
  W = weight_variable('%s_W'%nmn, s, tb)
  b = bias_variable('%s_b'%nmn, [s[3]], tb)
  
  h  = tf.nn.conv2d(x, W, strides=[1, strd, strd, 1], padding='SAME') + b;

  if bn:
    h = tf.layers.batch_normalization(h, name='%s_bn'%nmn, training=training)
  
  #h = tf.nn.relu(h)
  if alpha < 0:
    h = tf.nn.softplus(h, name='%s/softplus'%nmn)
  else:
    h = tf.nn.leaky_relu(h, alpha, name='%s/leaky_relu'%nmn)

  return h

def deconv2d(x, s, nmn, training=True, tb=True, bn=True, alpha=0.0):
  h = tf.layers.conv2d_transpose(x, s[0], [s[1], s[2]], strides=2, activation=tf.identity, padding='SAME', name=nmn, trainable=tb)

  if bn:
    h = tf.layers.batch_normalization(h, name='%s/bn'%nmn, training=training)
  
  if alpha < 0:
    h = tf.nn.softplus(h, name='%s/softplus'%nmn)
  else:
    h = tf.nn.leaky_relu(h, alpha, name='%s/leaky_relu'%nmn)

  return h

def max_pool_2x2(x, nmn):
  with tf.name_scope('pool1'):
    h = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  return h


def fc(x, s, nmn, activation=True, tb=True):
  W = weight_variable('%s_W'%nmn, s, tb)
  b = bias_variable('%s_b'%nmn, [s[1]], tb)

  h = tf.matmul(x, W) + b
  if activation:
    h = tf.nn.relu(h)

  return h


def weight_variable(nmn, shape, tb=True):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.get_variable(nmn, initializer=initial, trainable=tb)


def bias_variable(nmn, shape, tb=True):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.get_variable(nmn, initializer=initial, trainable=tb)