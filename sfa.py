import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
mpl.use('Agg')
import matplotlib.pyplot as plt
import mdp # mdp needed for data processing
from sklearn import preprocessing

max_time = 2 # time in seconds 
omega_s = 1 # slow signal frequency                                                                                                                                                                                                                                                                                                                                         
omega_m = 10 # fast signal frequency
omega_f = 20 # fast signal frequency                                                                                                                                                                                                                                                                                                                                        

N = 300 # nuber of samples
time = np.linspace(0, max_time, num=N) # timeline

# The speed of a particle is slow in one direction
slow_signal = np.cos(omega_s*2*np.pi*time)
med_signal = np.sin(omega_m*2*np.pi*time)
fast_signal = np.sin(omega_f*2*np.pi*time)


# Original trajectory
xt = fast_signal 
yt = med_signal
zt = slow_signal


# Plot the original trajectory:
plt.subplot(311) 
plt.plot(np.arange(N), xt, 'r', linewidth=0.7, label="slow signal")
plt.legend(loc='upper right')
plt.xlabel('time t')
plt.ylabel('$S$')
plt.subplot(312)
plt.plot(np.arange(N), yt,  label="medium signal")
plt.legend(loc='upper right')
plt.xlabel('time t')
plt.ylabel('$F$')
plt.subplot(313)
plt.plot(np.arange(N), zt,  label="fast signal")
plt.legend(loc='upper right')
plt.xlabel('time t')
plt.ylabel('$F$')
plt.show()                                                                                                                                                                                                                                                                                                                                                                 
plt.savefig('./original_signal.png')


# Trajectory in joint space
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(xt, yt, zt)
plt.show()

# Combine three signals in 3-tensor
X = np.ndarray(shape = (3,N))
X[0,] = xt
X[1,] = yt
X[2,] = zt
Xt = np.transpose(X)




# Rotate the sensors
theta = np.pi/5
phi = 5*np.pi/12
Rx = np.zeros(shape = (3,3))
Rx[0,0] = 1
Rx[1,1] = np.cos(theta)
Rx[1,2] = -np.sin(theta)
Rx[2,1] = np.sin(theta)
Rx[2,2] = np.cos(theta)
Ry = np.zeros(shape = (3,3))
Ry[1,1] = 1
Ry[0,0] = np.cos(phi)
Ry[0,2] = np.sin(phi)
Ry[2,0] = -np.sin(phi)
Ry[2,2] = np.cos(phi)
X = np.matmul(Ry,X)

# Plot the sensors measurements:
plt.subplot(311) 
plt.plot(np.arange(N), X[0,], 'r', linewidth=0.7, label="slow signal")
plt.legend(loc='upper right')
plt.xlabel('time t')
plt.ylabel('$S$')
plt.subplot(312)
plt.plot(np.arange(N), X[1,],  label="medium signal")
plt.legend(loc='upper right')
plt.xlabel('time t')
plt.ylabel('$F$')
plt.subplot(313)
plt.plot(np.arange(N), X[2,],  label="fast signal")
plt.legend(loc='upper right')
plt.xlabel('time t')
plt.ylabel('$F$')
plt.show()                                                                                                                                                                                                                                                                                                                                                                 
plt.savefig('./original_signal_1.png')

# Signal in joint space
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(X[0,], X[1,], X[2,])
plt.show()


# AIM: extract the original slow component from the measurments of rotated sensors

# Preprocess 1: Y is zero-mean, unit-variance X
Y = np.ndarray(shape = (3,N))
Y[0,] = preprocessing.scale((X[0,]))
Y[1,] = preprocessing.scale((X[1,]))
Y[2,] = preprocessing.scale((X[2,]))
Y = np.transpose(Y)

# Nonlinear expansion: Y_expanded, size is N_timepoints * (I + I(I+1)/2), I is dimension of input
nonlinear_expand = mdp.nodes.PolynomialExpansionNode(3) # nonlinear expansion
Y_expanded = nonlinear_expand(Y)

# Preprocess 2: Z is zero-mean, unit-variance Y_expanded
T=np.transpose(Y_expanded)
Z = np.ndarray(shape = (3,N))
Z[0,] = preprocessing.scale((np.transpose(T[0,])))
Z[1,] = preprocessing.scale((np.transpose(T[1,])))
Z[2,] = preprocessing.scale((np.transpose(T[2,])))
Z = np.transpose(Z)

# Temporal derivative
derivative = lambda X: X[1:, :]-X[:-1, :]

U, S, V = np.linalg.svd(derivative(Z), full_matrices=False)
plt.subplot(1, 2, 2)
Q = Y[:-1].dot(V[-1,:])
plt.plot(Q)

#tf = mdp.nodes.TimeFramesNode(4)
#tfX = tf.execute(X)
#cubic_expand = mdp.nodes.PolynomialExpansionNode(3)
#cubic_expanded_X = cubic_expand(tfX)