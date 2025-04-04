from config import *
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

import shape as sh
import rotation_functions as rf

# Prepare pseudo-observation angles:
#------------------------------------------------------------
# Find a series of N_obs_angles rotation matrices:
N_obs_angles = 100
x, y, z = sh.points_to_sphere(N_obs_angles * 2)

# Filter out all the observation angles which are symmetrical:
hemisphere = np.where(z>=0)[0]
x, y, z = x[hemisphere], y[hemisphere], z[hemisphere]
#------------------------------------------------------------

# Decide on some useful pointings:
#------------------------------------------------------------
# Face-down on the x-y orbital plane:
xyz_fd = [0.,0.,1.]

# Edge-on to the x-y orbital plane, pointing along the tail:
xyz_eoa = [1.,0.,0.]

# Edge-on to the x-y orbital plane, pointing normal to the tail:
xyz_eob = [0.,1.,0.]
#------------------------------------------------------------

# Find the corresponding rotation matrices:
#------------------------------------------------------------
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

angles = np.array([angle_between([l,n,m], xyz_fd) for l,n,m in zip(x,y,z)])
angle_fd = np.sin(angles).argmin()
print('Index for face-on: %i, off by %.2f degrees' % (angle_fd, angles[angle_fd]*180/np.pi))

angles = np.array([angle_between([l,n,m], xyz_eoa) for l,n,m in zip(x,y,z)])
angle_eoa = np.sin(angles).argmin()
print('Index for edge-on along tail: %i, off by %.2f degrees' % (angle_eoa, angles[angle_eoa]*180/np.pi))

angles = np.array([angle_between([l,n,m], xyz_eob) for l,n,m in zip(x,y,z)])
angle_eob = np.sin(angles).argmin()
print('Index for face-on normal to tail: %i, off by %.2f degrees' % (angle_eob, angles[angle_eob]*180/np.pi))
#------------------------------------------------------------

cull = [i not in [angle_fd, angle_eoa, angle_eob] for i in np.arange(len(x))]

import mpl_toolkits.mplot3d
plt.figure().add_subplot(111, projection='3d').scatter(x[cull], y[cull], z[cull]);
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.gca().set_zlim(-1,1)
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_zlabel('z')

for angle, color in zip([angle_fd, angle_eoa, angle_eob], ['r', 'g', 'b']):
  plt.gca().scatter(x[angle], y[angle], z[angle], color=color)

plt.gca().plot([0,1],[0,0],[0,0], 'g')
plt.gca().plot([0,0],[0,-1],[0,0], 'b')
plt.gca().plot([0,0],[0,0],[0,1], 'r')
