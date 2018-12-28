# Multiple Operations on a Computational Graph

# In this script, we will create an array and perform two multiplications on it, 
# followed by addition:output = (input) * (m1) * (m2) + (a1)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import os

ops.reset_default_graph()

# Create Session
sess = tf.Session()

# Create data to feed in
my_array = np.array([[1.,3.,5.,7.,9.],
                  [-2.,0.,2.,4.,6.],
                  [-6.,-3.,0.,3.,6.]])
# Duplicate the array for having two inputs
x_vals = np.array([my_array,my_array + 1])
# Declare the placeholder
x_data = tf.placeholder(tf.float32,shape=(3,5))
# Declare constants for operations
m1 = tf.constant([[1.],[0.],[-1.],[2.],[4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

# print 
print('x_vals:',x_vals)
print('m1',m1)
print('m2:',m2)
print('a1:',a1)

# 1st Operation Layer = Multiplication
prod1 = tf.matmul(x_data,m1)
# 2nd Operation Layer = Multiplication
prod2 = tf.matmul(prod1,m2)
# 3rd Operation Layer = Addition
add1 = tf.add(prod2,a1)

# Evaluate and Print Output
print("result")
for x_val in x_vals:
    print(sess.run(add1,feed_dict={x_data:x_val}))

# Create and Format Tensorboard outputs for viewing
merget = tf.summary.merge_all(key='summaries')

#if not os.path.exists('2_tensorboard_logs/'):
#    os.makedirs('2_tensorboard_logs/')

my_writer = tf.summary.FileWriter('2_tensorboard_logs/',sess.graph)