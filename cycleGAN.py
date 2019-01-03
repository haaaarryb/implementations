import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tfl
from cadl.cycle_gan import lrelu, instance_norm

def encoder(x, n_filters=32, k_size=3, normaliser_fn=instance_norm, activation_fn=lrelu, scope=None, reuse=None):
    with tf.variable_scope(scope or 'ecoder', reuse=reuse):
        h = tf.pad(x, [[0, 0], [k_size, k_size], [k_size, k_size], [0, 0]], 'REFLECT')
        h = tfl.conv2d(inputs=h,
                       num_outputs=n_filters,
                       kernel_size=7,
                       stride=1,
                       padding='VALID',
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                       biases_initializer=None,
                       normalizer_fn=normaliser_fn,
                       activation_fn=activation_fn,
                       scope='1',
                       reuse=reuse)
        h = tfl.conv2d(inputs=h,
                       num_outputs=n_filters * 2,
                       kernel_size=k_size,
                       stride=2,
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                       biases_initializer=None,
                       normalizer_fn=normaliser_fn,
                       activation_fn=activation_fn,
                       scope='2',
                       reuse=reuse)
        h = tfl.conv2d(inputs=h,
                       num_outputs=n_filters * 4,
                       kernel_size=k_size,
                       stride=2,
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                       biases_initializer=None,
                       normalizer_fn=normaliser_fn,
                       activation_fn=activation_fn,
                       scope='3',
                       reuse=reuse)
        return h

def res_block(x, n_channels=128, normaliser_fn=instance_norm,
              activation_fn=lrelu, k_size=3, scope=None, reuse=None):
    with tf.variable_scope(scope or 'residual', reuse=reuse):
        h = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        h = tfl.conv2d(inputs=h,
                       num_outputs=n_channels* 4,
                       kernel_size=k_size,
                       stride=2,
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                       biases_initializer=None,
                       normalizer_fn=normaliser_fn,
                       activation_fn=activation_fn,
                       padding='VALID',
                       scope='3',
                       reuse=reuse)
        h = tf.add(x, h)
    return h

def transform(x, img_size=256, reuse=None):
    h = x
    n_blocks = 9
    for block_idx in range(n_blocks):
        with tf.variable_scope('block{}'.format(block_idx), reuse=reuse):
            h = res_block(h, reuse=reuse)
    return h

def decoder(x, n_filters=32, k_size=5, normaliser_fn=instance_norm,
            activation_fn=lrelu, scope=None, reuse=None):
    with tf.variable_scope(scope or 'decoder', reuse=reuse):
        h = tfl.conv2d_transpose(inputs=x,
                       num_outputs=n_filters * 2,
                       kernel_size=k_size,
                       stride=2,
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                       biases_initializer=None,
                       normalizer_fn=normaliser_fn,
                       activation_fn=activation_fn,
                       scope='1',
                       reuse=reuse)
        h = tfl.conv2d_transpose(inputs=h,
                       num_outputs=n_filters,
                       kernel_size=k_size,
                           stride=2,
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                       biases_initializer=None,
                       normalizer_fn=normaliser_fn,
                       activation_fn=activation_fn,
                       scope='2',
                       reuse=reuse)
        h = tfl.conv2d_transpose(inputs=h,
                       num_outputs=3,
                       kernel_size=7,
                       stride=1,
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                       biases_initializer=None,
                       normalizer_fn=normaliser_fn,
                       activation_fn=tf.nn.tanh,            #try sigmoid
                       scope='3',
                       reuse=reuse)
        return h

def generator(x, scope=None, reuse=None):
    img_size= x.get_shape().as_list()[1]
    with tf.variable_scope(scope or 'generator', reuse=reuse):
        h = encoder(x, reuse=reuse)
        h = transform(h, img_size, reuse=reuse)
        h = decoder(h, reuse=reuse)
    return h

def discriminator(x, n_filters=64, k_size=4, activation_fn=lrelu,
                  normaliser_fn=instance_norm, scope=None, reuse=None):
    with tf.variable_scope(scope or 'discriminator', reuse=reuse):
        h = tfl.conv2d(inputs=x,
                       num_outputs=n_filters,
                       kernel_size=k_size,
                       stride=2,
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                       biases_initialiser=None,
                       activation_fn=activation_fn,
                       normaliser_fn=None,
                       scope='1',
                       reuse=reuse)
        h = tfl.conv2d(inputs=h,
                       num_outputs=n_filters * 2,
                       kernel_size=k_size,
                       stride=2,
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                       biases_initialiser=None,
                       activation_fn=activation_fn,
                       normaliser_fn=normaliser_fn,
                       scope='2',
                       reuse=reuse)
        h = tfl.conv2d(inputs=h,
                       num_outputs=n_filters * 4,
                       kernel_size=k_size,
                       stride=2,
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                       biases_initialiser=None,
                       activation_fn=activation_fn,
                       normaliser_fn=normaliser_fn,
                       scope='2',
                       reuse=reuse)
        h = tfl.conv2d(inputs=h,
                    num_outputs=n_filters * 4,
                    kernel_size=k_size,
                    stride=2,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    biases_initializer=None,
                    activation_fn=activation_fn,
                    normalizer_fn=normalizer_fn,
                    scope='3',
                    reuse=reuse)
        h = tfl.conv2d(inputs=h,
                    num_outputs=n_filters * 8,
                    kernel_size=k_size,
                    stride=1,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    biases_initializer=None,
                    activation_fn=activation_fn,
                    normalizer_fn=normalizer_fn,
                    scope='4',
                    reuse=reuse)
        h = tfl.conv2d(
                    inputs=h,
                    num_outputs=1,
                    kernel_size=k_size,
                    stride=1,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    biases_initializer=None,
                    activation_fn=tf.nn.sigmoid,
                    scope='5',
                    reuse=reuse)
        return h


img_size = 256
X_real = tf.placeholder(name='X', shape=[1, img_size, img_size, 3], dtype=tf.float32)
Y_real = tf.placeholder(name='Y', shape=[1, img_size, img_size, 3], dtype=tf.float32)

X_fake = generator(Y_real, scope='G_yx')
Y_fake = generator(X_real, scope='G_xy')

X_cycle = generator(Y_fake, scope='G_yx', reuse=True)
Y_cycle = generator(X_fake, scope='G_xy', reuse=True)

D_X_real = discriminator(X_real, scope='D_X')
D_Y_real = discriminator(Y_real, scope='D_Y')
D_X_fake = discriminator(X_fake, scope='D_X', reuse=True)
D_Y_fake = discriminator(Y_fake, scope='D_Y', reuse=True)

l1 = 10.0
loss_cycle = tf.reduce_mean(l1 * tf.abs(X_real - X_cycle)) + \
             tf.reduce_mean(l1 * tf.abs(Y_real - Y_cycle))
loss_G_xy = tf.reduce_mean(tf.square(D_Y_fake - 1.0)) + loss_cycle
loss_G_yx = tf.reduce_mean(tf.square(D_X_fake - 1.0)) + loss_cycle

X_fake_sample = tf.placeholder(name='X_fake_sample', shape=[None, img_size, img_size, 3], dtype=tf.float32)
Y_fake_sample = tf.placeholder(name='Y_fake_sample', shape=[None, img_size, img_size, 3], dtype=tf.float32)

D_X_fake_sample = discriminator(X_fake_sample, scope='D_X', reuse=True)
D_Y_fake_sample = discriminator(Y_fake_sample, scope='D_Y', reuse=True)

loss_D_Y = (tf.reduce_mean(tf.square(D_Y_real - 1.0)) + \
            tf.reduce_mean(tf.square(D_Y_fake_sample))) / 2.0
loss_D_X = (tf.reduce_mean(tf.square(D_X_real - 1.0)) + \
            tf.reduce_mean(tf.square(D_X_fake_sample))) / 2.0

tf.reset_default_graph()
from cadl.cycle_gan import cycle_gan
net = cycle_gan(img_size=img_size)

training_vars = tf.trainable_variables()
D_X_vars = [v for v in training_vars if v.name.startswith('D_X')]
D_Y_vars = [v for v in training_vars if v.name.startswith('D_Y')]
G_xy_vars = [v for v in training_vars if v.name.startswith('G_xy')]
G_yx_vars = [v for v in training_vars if v.name.startswith('G_yx')]

learning_rate = 0.001
D_X = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net['loss_D_X'], var_list=D_X_vars)
D_Y = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net['loss_D_Y'], var_list=D_Y_vars)
G_xy = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net['loss_G_xy'], var_list=G_xy_vars)
G_yx = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net['loss_G_yx'], var_list=G_yx_vars)

# GAN might forget to call out old fake images as distributions change
# keep some old fakes around to stabilise training and keep training dist representative of all fake imgs
capacity = 50 # number of fake generations to keep around

# Storage for fake generations
fake_Xs = capacity * [np.zeros((1, img_size, img_size, 3), dtype=np.float32)]
fake_Ys = capacity * [np.zeros((1, img_size, img_size, 3), dtype=np.float32)]