import numpy as np

def points_to_sphere(N):
  # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
  indices = np.arange(0, N) + 0.5
  phi = np.arccos(1 - 2*indices/N)
  theta = np.pi * (1 + 5**0.5) * indices
  x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
  return x, y, z

def rotation_matrix_from_vectors(vec1, vec2):
  # From https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
  """ Find the rotation matrix that aligns vec1 to vec2
  :param vec1: A 3d "source" vector
  :param vec2: A 3d "destination" vector
  :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
  """
  a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
  v = np.cross(a, b)
  c = np.dot(a, b)
  s = np.linalg.norm(v)
  kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
  return rotation_matrix

almnt = lambda E: np.arccos(np.dot(np.dot(E,[1.,0.]),[1.,0.]))

def Ellipsoid(pos, a, R):
    x = np.dot(R.T, pos.T)
    return np.sum(np.divide(x.T, a)**2, axis=1)

def shape(sim, cut, nbins=100, rmin=None, rmax=None, weight='mass', bins='equal', ndim=3, max_iterations=10, tol=1e-3, justify=False):

  """
  Calculates the shape of sim in homeoidal shells, over a range of nbins radii.
  Homeoidal shells maintain a fixed area (ndim=2) or volume (ndim=3).
  The algorithm is sensitive to substructure, which should ideally be removed.
  Particles must be in a centered frame.
  Caution is advised when assigning large number of bins and radial
  ranges with many particles, as the algorithm becomes very slow.

  **Input:**

  *nbins* (default=100): The number of homeoidal shells to consider.

  *rmin* (default=None): The minimum initial radial bin in units of
  ``sim['pos']``. By default this is taken as rmax/1000.

  *rmax* (default=None): The maximum initial radial bin in units of
  ``sim['pos']``. By default this is taken as the greatest radial value.

  *bins* (default='equal'): The spacing scheme for the homeoidal shell bins.
  ``equal`` initialises radial bins with equal numbers of particles.
  This number is not necessarily maintained during fitting.
  ``log`` and ``lin`` initialise bins with logarithmic and linear
  radial spacing.

  *ndim* (default=3): The number of dimensions to consider. If ndim=2,
  the shape is calculated in the x-y plane. The user is advised to make
  their own cut in ``z`` if using ndim=2.

  *max_iterations* (default=10): The maximum number of shape calculations.
  Fewer iterations will result in a speed-up, but with a bias towards
  increadingly spheroidal shape calculations.

  *tol* (default=1e-3): Convergence criterion for the shape calculation.
  Convergence is achieved when the axial ratios have a fractional
  change <=tol between iterations

  *justify* (default=False): Align the rotation matrix directions
  such that they point in a singular consistent direction aligned
  with the overall halo shape. This can be useful if working with slerps.

  **Output**:

  *rbins*: The radii of the initial spherical bins in units
  of ``sim['pos']``.

  *axis lengths*: The axis lengths of each homoeoidal shell in
  order a>b>c with units of ``sim['pos']``.

  *N*: The number of particles within each bin.

  *R*: The rotation matrix of each homoeoidal shell.

  """

  # Sanitise inputs:
  if (rmax == None): rmax = sim['r'][cut].max()*1.0001
  if (rmin == None): rmin = rmax/1E3
  assert ndim in [2, 3]
  assert max_iterations > 0
  assert tol > 0
  assert rmin >= 0
  assert rmax > rmin
  assert nbins > 0
  if ndim==2:
    assert np.sum((sim['r2'][cut] >= rmin) & (sim['r2'][cut] < rmax)) > nbins*2
  elif ndim==3:
    assert np.sum((sim['r'][cut] >= rmin) & (sim['r'][cut] < rmax)) > nbins*2
  if bins not in ['equal', 'log', 'lin']: bins = 'equal'

  # Handy 90 degree rotation matrices:
  Rx = np.array([[1,0,0], [0,0,-1], [0,1,0]])
  Ry = np.array([[0,0,1], [0,1,0], [-1,0,0]])
  Rz = np.array([[0,-1,0], [1,0,0], [0,0,1]])

  #-----------------------------FUNCTIONS-----------------------------
  sn = lambda r,N: np.append([r[i*int(len(r)/N):(1+i)*int(len(r)/N)][0]\
                              for i in range(N)],r[-1])

  # General equation for an ellipse/ellipsoid:
  def Ellipsoid(pos, a, R):
      x = np.dot(R.T, pos.T)
      return np.sum(np.divide(x.T, a)**2, axis=1)

  # Define moment of inertia tensor:
  def MoI(r, m, ndim=3):
    return np.array([[np.sum(m*r[:,i]*r[:,j]) for j in range(ndim)] for i in range(ndim)])

  # Calculate the shape in a single shell:
  def shell_shape(r,pos,mass, a,R, r_range, ndim=3):

    # Find contents of homoeoidal shell:
    mult = r_range / np.mean(a)
    in_shell = (r > min(a)*mult[0]) & (r < max(a)*mult[1])
    pos, mass = pos[in_shell], mass[in_shell]
    inner = Ellipsoid(pos, a*mult[0], R)
    outer = Ellipsoid(pos, a*mult[1], R)
    in_ellipse = (inner > 1) & (outer < 1)
    ellipse_pos, ellipse_mass = pos[in_ellipse], mass[in_ellipse]

    # End if there is no data in range:
    if not len(ellipse_mass):
      return a, R, np.sum(in_ellipse)

    # Calculate shape tensor & diagonalise:
    D = list(np.linalg.eigh(MoI(ellipse_pos,ellipse_mass,ndim) / np.sum(ellipse_mass)))

    # Rescale axis ratios to maintain constant ellipsoidal volume:
    R2 = np.array(D[1])
    a2 = np.sqrt(abs(D[0]) * ndim)
    div = (np.prod(a) / np.prod(a2))**(1/float(ndim))
    a2 *= div

    return a2, R2, np.sum(in_ellipse)

  # Re-align rotation matrix:
  def realign(R, a, ndim):
    if ndim == 3:
      if a[0]>a[1]>a[2]<a[0]: pass                          # abc
      elif a[0]>a[1]<a[2]<a[0]: R = np.dot(R,Rx)            # acb
      elif a[0]<a[1]>a[2]<a[0]: R = np.dot(R,Rz)            # bac
      elif a[0]<a[1]>a[2]>a[0]: R = np.dot(np.dot(R,Rx),Ry) # bca
      elif a[0]>a[1]<a[2]>a[0]: R = np.dot(np.dot(R,Rx),Rz) # cab
      elif a[0]<a[1]<a[2]>a[0]: R = np.dot(R,Ry)            # cba
    elif ndim == 2:
      if a[0]>a[1]: pass                                    # ab
      elif a[0]<a[1]: R = np.dot(R,Rz[:2,:2])               # ba
    return R

  # Calculate the angle between two vectors:
  def angle(a, b):
    return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

  # Flip x,y,z axes of R2 if they provide a better alignment with R1.
  def flip_axes(R1, R2):
    for i in range(len(R1)):
      if angle(R1[:,i], -R2[:,i]) < angle(R1[:,i], R2[:,i]):
        R2[:,i] *= -1
    return R2
  #-----------------------------FUNCTIONS-----------------------------

  # Set up binning:
  r = np.array(sim['r'][cut]) if ndim==3 else np.array(sim['r2'][cut])
  pos = np.array(sim['pos'][cut])[:,:ndim]
  mass = np.array(sim[weight][cut])

  if (bins == 'equal'): # Bins contain equal number of particles
      full_bins = sn(np.sort(r[(r>=rmin) & (r<=rmax)]), nbins*2)
      bin_edges = full_bins[0:nbins*2+1:2]
      rbins = full_bins[1:nbins*2+1:2]
  elif (bins == 'log'): # Bins are logarithmically spaced
      bin_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
      rbins = np.sqrt(bin_edges[:-1] * bin_edges[1:])
  elif (bins == 'lin'): # Bins are linearly spaced
      bin_edges = np.linspace(rmin, rmax, nbins+1)
      rbins = 0.5*(bin_edges[:-1] + bin_edges[1:])

  # Initialise the shape arrays:
  rbins = rbins
  axial_lengths = np.zeros([nbins,ndim])
  N_in_bin = np.zeros(nbins).astype('int')
  rotations = [0]*nbins

  # Loop over all radial bins:
  for i in range(nbins):

    # Initial spherical shell:
    a = np.ones(ndim) * rbins[i]
    a2 = np.zeros(ndim)
    a2[0] = np.inf
    R = np.identity(ndim)

    # Iterate shape estimate until a convergence criterion is met:
    iteration_counter = 0
    while ((np.abs(a[1]/a[0] - np.sort(a2)[-2]/max(a2)) > tol) & \
           (np.abs(a[-1]/a[0] - min(a2)/max(a2)) > tol)) & \
           (iteration_counter < max_iterations):
      a2 = a.copy()
      a,R,N = shell_shape(r,pos,mass, a,R, bin_edges[[i,i+1]], ndim)
      iteration_counter += 1

    # Adjust orientation to match axis ratio order:
    R = realign(R, a, ndim)

    # Ensure consistent coordinate system:
    if np.sign(np.linalg.det(R)) == -1:
      R[:,1] *= -1

    # Update profile arrays:
    a = np.flip(np.sort(a))
    axial_lengths[i], rotations[i], N_in_bin[i] = a, R, N

  # Ensure the axis vectors point in a consistent direction:
  if justify:
    _, _, _, R_global = shape(sim, nbins=1, rmin=rmin, rmax=rmax, ndim=ndim)
    rotations = np.array([flip_axes(R_global, i) for i in rotations])

  return rbins, bin_edges, np.squeeze(axial_lengths.T).T, N_in_bin, np.squeeze(rotations)
