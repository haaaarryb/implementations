import tensorflow as tf
tf.enable_eager_execution()

x = tf.zeros([10, 10])
print(x)
x += 2
print(x)

v = tf.Variable(1.0)
assert v.numpy() == 1.0

v.assign(3.0)
assert v.numpy() == 3.0

v.assign(tf.square(v))
assert v.numpy() == 9.0

class Model(object):
    def __init__(self):
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b

model = Model()

assert model(3.0).numpy() == 15.0

def loss(pred, label):
    return tf.reduce_mean(tf.square(pred - label))

TRUE_W = 3.0
TRUE_B = 2.0
M = 1000

inputs = tf.random_normal(shape=[M])
noise = tf.random_normal(shape=[M])
outputs = inputs * TRUE_W + TRUE_B + noise

print(inputs.shape)
print(outputs.shape)

import matplotlib.pyplot as plt
plt.scatter(inputs, outputs, c='r')
plt.scatter(inputs, model(inputs), c='b')
plt.show()
