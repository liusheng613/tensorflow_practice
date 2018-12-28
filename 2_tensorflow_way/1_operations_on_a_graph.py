# In this script, we create an array and feed it into a placeholder. 
#  We then multiply it by a constant.
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.framework import ops

ops.reset_default_graph()

# Create Session
sess = tf.Session()

# Create tensors
# Create data to feed in
x_vals = np.array([1.,3.,5.,7.,9.])
x_data = tf.placeholder(tf.float32)
m = tf.constant(3.)

print('x_vals:',x_vals)
print('m:',m)

# Multiplication
prod = tf.multiply(x_data,m)
print('m multiply x_val:')
for x_val in x_vals:
    print(sess.run(prod,feed_dict={x_data:x_val}))

merged = tf.summary.merge_all(key='summaries')
#if not os.path.exists('1_tensorboard_logs/'):
#    os.makedirs('1_tensorboard_logs/')

my_writer = tf.summary.FileWriter('1_tensorboard_logs/',sess.graph)
my_writer.flush() 
my_writer.close()
