import tensorflow as tf
import numpy as np

tf.enable_eager_execution()
if tf.test.is_gpu_available():
    print('I can use a GPU.')
else:
    print('I cannot use a GPU.')

ds_tensors = tf.data.Dataset.from_tensor_slices([[1, 2, 3, 4, 5]])
print(ds_tensors)

# apply transforms
ds_tensors = ds_tensors.map(tf.square)
ds_tensors = ds_tensors.shuffle(2)
ds_tensors = ds_tensors.batch(2)

print(ds_tensors)
for x in ds_tensors:
    print(x)
    print(x.dtype)
    print(x.shape)
    print()

import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
    f.write("""Line 1
Line 2
Line 3""")

ds_file = tf.data.TextLineDataset(filename)
print(ds_file)
for i in ds_file:
    print(i)
k

ds_file = ds_file.batch(2)



