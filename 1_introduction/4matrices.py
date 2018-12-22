# Matrices and Matrix Operations
#----------------------------------
#
# This function introduces various ways to create
# matrices and how to use them in TensorFlow

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()


sess = tf.Session()


# Declaring matrices

# Identity matrix
identity_matrix = tf.diag([1.0,1.0,1.0])
print('identity_matrix:')
print(sess.run(identity_matrix))
print()

# 2x3 random norm matrix
A = tf.truncated_normal([2,3])
print('A:')
print(sess.run(A))
print()

# 2x3 constant matrix
B = tf.fill([2,3],5.0)
print('B:')
print(sess.run(B))
print()

# 3x2 random uniform matrix
C = tf.random_uniform([3,2])
print('C:')
print(sess.run(C))
print('C again:')
print(sess.run(C)) # Note that we are reinitializing, hence the new random variabels
print()

# Create matrix from np array
D = tf.convert_to_tensor(np.array([[1.,2.,3.],[-3.,-7.,-1.],[0.,5.,-2.]]))
print('D:')
print(sess.run(D))
print()

# Matrix addition/subtraction
print('A+B:')
print(sess.run(A+B))
print('A-B:')
print(sess.run(A-B))
print()

# Matrix Multiplication
print('B x identity_matrix:')
print(sess.run(tf.matmul(B,identity_matrix)))
print()

# Matrix Transpose
print('transpose C:')
print(sess.run(tf.transpose(C))) # Again, new random variables
print()

# Matrix Determinant
print('determinant D:')
print(sess.run(tf.matrix_determinant(D)))
print()

# Matrix Inverse
print('inverse D:')
print(sess.run(tf.matrix_inverse(D)))
print()

# Cholesky Decomposition
print('cholesky identity_matrix:')
print(sess.run(tf.cholesky(identity_matrix)))
print()

# Eigenvalues and Eigenvectors
print('adjoint_eig D:')
print(sess.run(tf.self_adjoint_eig(D)))
print()