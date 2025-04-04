import numpy as np

def R_x(theta):
  return np.array([[1.,0.,0.], \
                   [0.,np.cos(theta),-np.sin(theta)], \
                   [0.,np.sin(theta),np.cos(theta)]])
def R_y(theta):
  return np.array([[np.cos(theta),0.,np.sin(theta)], \
                   [0.,1.,0.], \
                   [-np.sin(theta),0.,np.cos(theta)]])
def R_z(theta):
  return np.array([[np.cos(theta),-np.sin(theta),0.], \
                   [np.sin(theta),np.cos(theta),0.], \
                   [0.,0.,1.]])
def angle(a, b):
  # Angle between two vectors:
  return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def align_orbit(pos, vel):

  '''
  Align the vector "pos" to point along the vector "vel".
  '''

  theta_xy = angle(pos[[0, 1]], [0., 1.])
  rotate_z = R_z(np.sign(pos[0]) * theta_xy)
  pos = np.dot(rotate_z, pos)
  vel = np.dot(rotate_z, vel)

  theta_yz = angle(pos[[1, 2]], [1., 0.])
  rotate_x = R_x(-np.sign(pos[2]) * theta_yz)
  pos = np.dot(rotate_x, pos)
  vel = np.dot(rotate_x, vel)

  theta_xz = angle(vel[[0, 2]], [1., 0.])
  rotate_y = R_y(np.sign(vel[2]) * theta_xz)

  return np.dot(rotate_y, np.dot(rotate_x, rotate_z))

