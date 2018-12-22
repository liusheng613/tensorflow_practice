# Operations
#----------------------------------
#
# This function introduces various operations
# in TensorFlow

# Declaring Operations
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Open graph session
sess = tf.Session()

# div() vs truediv() vs floordiv()
print('div(3,4):',end='')
print(sess.run(tf.div(3,4)))

print('truediv(3,4):',end='')
print(sess.run(tf.truediv(3,4)))

print('floordiv(3.0,4.0):',end='')
print(sess.run(tf.floordiv(3.0,4.0)))

# Mod function
print('mod(22.0,5.0):',end='')
print(sess.run(tf.mod(22.0,5.0)))

# Cross Product
print('cross([1.,0.,0.],[0.,1.,0.]:',end='')
print(sess.run(tf.cross([1.,0.,0.],[0.,1.,0.])))

# Trig functions
print('sin(3.1416):',end='')
print(sess.run(tf.sin(3.1416)))
print('cos(3.1416):',end='')
print(sess.run(tf.cos(3.1416)))

# Tangent
print('Tangent 3.1416/4.:',end='')
print(sess.run(tf.div(tf.sin(3.1416/4.),tf.cos(3.1416/4.))))

# Custom operation
test_nums = range(15)
print('test_nums:',end='')
print(test_nums)


def custom_polynomial(x_val):
    # Return 3x^2 -x + 10
    return (tf.subtract(3 * tf.square(x_val),x_val) + 10)
print('3x^2 -x + 10,x=11:')
print(sess.run(custom_polynomial(11)))

# What should we get with list comprehension
expected_output = [3*x*x*x - x + 10 for x in test_nums]
print('3*x*x*x-x+10,x in test_nums in list way:')
print(expected_output)

# TensorFlow custom function output
print('3*x*x*x-x+10,x in test_nums using custom_polynomial fuction:')
for num in test_nums:
    print(sess.run(custom_polynomial(num)))