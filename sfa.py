import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
#mpl.use('Agg')
import matplotlib.pyplot as plt
import mdp # mdp needed for data processing
from sklearn  import preprocessing
import math 


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    
    
def plot_trajectory(X):
    """
    Plots the trajectory of the feature measured by orthogonal sensors
    """
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(X[0,], X[1,], X[2,])
    plt.show()
    
def plot_components(X):
    """
    Plots measurements of three orthogonal sensors
    """
    plt.subplot(311) 
    plt.plot(np.arange(N), X[0,], 'r', linewidth=0.7, label="x-signal")
    plt.legend(loc='upper right')
    plt.xlabel('time t')
    plt.ylabel('$S$')
    plt.subplot(312)
    plt.plot(np.arange(N), X[1,],  label="y-signal")
    plt.legend(loc='upper right')
    plt.xlabel('time t')
    plt.ylabel('$M$')
    plt.subplot(313)
    plt.plot(np.arange(N), X[2,],  label="z-signal")
    plt.legend(loc='upper right')
    plt.xlabel('time t')
    plt.ylabel('$F$')
    plt.show()

def whiten(X):
    Y = np.ndarray(shape = (X.shape[0],N))
    for i in range(X.shape[0]): 
        Y[i,] = preprocessing.scale((X[i,]))
    #Y = np.transpose(Y)
    return Y


# Temporal derivative
derivative = lambda X: X[1:, :]-X[:-1, :]
# Nonlinear expansion
nonlinear_expand = mdp.nodes.PolynomialExpansionNode(2) # nonlinear expansion


max_time = 2 # time in seconds 
omega_s = 1 # slow signal frequency                                                                                                                                                                                                                                                                                                                                         
omega_m = 10 # medium-fast signal frequency
omega_f = 20 # fast signal frequency                                                                                                                                                                                                                                                                                                                                        

N = 300 # nuber of timepoints
time = np.linspace(0, max_time, num=N) # timeline

# Original measurements 
xt = np.sin(omega_f*2*np.pi*time) # fast signal
yt = np.sin(omega_m*2*np.pi*time) # medium-fast signal
zt = np.cos(omega_s*2*np.pi*time) # slow signal

# Combine three signals in 3-tensor
Xt = np.ndarray(shape = (3,N))
Xt[0,] = xt
Xt[1,] = yt
Xt[2,] = zt
Xt = whiten(Xt)
# Plot the sensors measurements:
print('Original sensors measurments:')
plot_components(Xt)
# Plot trajectory in joint space
print('Original feature trajectory:')
plot_trajectory(Xt)

# Rotate the sensors around axis by theta
axis = [2,2,4]
theta = 2*np.pi/7
X=np.matmul(rotation_matrix(axis, theta),Xt)
# Plot the sensors measurements:
print('Rotated sensors measurments:')
plot_components(X)
# Plot trajectory in joint space
print('Rotated feature trajectory:')
plot_trajectory(X)

# AIM: extract the original slow component from the measurments of rotated sensors
# normalize and shift X: mean(X)=0, var(X)=1
X = whiten(X)
X = np.transpose(X)
# Nonlinear expansion: Y_expanded, size is N_timepoints * (I + I(I+1)/2), I is dimension of input
Z = nonlinear_expand(X)
Z = np.transpose(whiten(np.transpose(Z)))

U, S, V = np.linalg.svd(derivative(Z), full_matrices=False)
plt.subplot(1, 2, 2)
Q = Z.dot(V[-1,:])
print('Extracted slowest feature:L')
plt.plot(Q)