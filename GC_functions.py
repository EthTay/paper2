import numpy as np

# Misc functions =======================================================
# Reformat scientific notation:
def latex_float(f):
    float_str = "{0:.1e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

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
  return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Rebin an array with a new resolution:
def rebin(x, y, new_x_bins, operation='mean'):

  index_radii = np.digitize(x, new_x_bins)
  if operation is 'mean':
    new_y = np.array([y[index_radii == i].mean() for i in range(1, len(new_x_bins))])
  elif operation is 'sum':
    new_y = np.array([y[index_radii == i].sum() for i in range(1, len(new_x_bins))])
  else:
    raise ValueError("Operation not supported")

  new_x = (new_x_bins[1:] + new_x_bins[:-1]) / 2
  new_y[np.isnan(new_y)] = np.finfo(float).eps

  return new_x, new_y

# Align orbit:
def align_orbit(pos, vel):
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
#=======================================================================

# Profile functions ====================================================
def Dehnen_profile(r, log_rs, log_Mg, gamma):
  rs = 10**log_rs
  Mg = 10**log_Mg
  rho_0 = Mg * (3. - gamma) / (4. * np.pi)
  return rho_0 * rs / ((r**gamma) * (r + rs)**(4 - gamma))

def Dehnen_mass(r, rs, Mg, gamma):
  return Mg * (r/(r+rs))**(3-gamma)

# Define the circular velocity in a Dehnen profile [Dehnen 1993]:
def Dehnen_vcirc(r, rs, Mg, gamma, G):
  return np.sqrt(G * Mg * r**(2 - gamma) / (r + rs)**(3 - gamma))
#=======================================================================

# Nbody6 functions =====================================================
# Shrink sphere centre:
'''
def shrink(s, shrink_factor=0.9, fraction=0.5):
  filt_orig = (s['kstara'] != 13) * (s['kstara'] != 14) * (s['mass'] < 10)
  pos = s['pos']*1
  r = np.linalg.norm(pos, axis=1)
  sphere = r[filt_orig].max()
  cen = np.zeros(3)
  filt = filt_orig.copy()
  while np.sum(r[filt] <= sphere) > max(fraction*s['nbound'][filt_orig].sum(), 100):
    CoM = np.average(pos[filt], weights=s['mass'][filt], axis=0)
    cen += CoM
    pos -= CoM
    sphere *= shrink_factor
    r = np.linalg.norm(pos, axis=1)
    filt *= r <= sphere
  return cen
'''
def shrink(s, shrink_factor=0.9, fraction=0.5):
  filt_orig = (s['kstara'] != 13) * (s['kstara'] != 14) * (s['mass'] < 10) * s['nbound']
  if not np.sum(filt_orig):
    filt_orig = s['nbound']
  cen = np.average(s['pos'][filt_orig], weights=s['mass'][filt_orig], axis=0)
  pos = s['pos'] - cen
  r = np.linalg.norm(pos, axis=1)
  sphere = r[filt_orig].max()
  filt = filt_orig.copy()
  while np.sum(r[filt] <= sphere) > max(fraction*filt_orig.sum(), 100):
    CoM = np.average(pos[filt], weights=s['mass'][filt], axis=0)
    cen += CoM
    pos -= CoM
    sphere *= shrink_factor
    r = np.linalg.norm(pos, axis=1)
    filt *= r <= sphere
  return cen

# Half radius:
def R_half(s, type='mass', filt=[0., np.inf], bound=True):
  r = s['r']
  body = s['nbound'] if bound else np.ones_like(s['nbound']).astype('bool')
  BHs = np.array([kstara in [13, 14] for kstara in s['kstara']])
  mass = s[type]
  filt = (mass <= filt[1]) & (mass >= filt[0])

  r, mass = r[~BHs & filt & body], mass[~BHs & filt & body]

  mass = mass[r.argsort()]
  r = np.sort(r)
  M_half = mass.sum() / 2.
  M_cum = np.cumsum(mass)
  return np.interp(M_half, M_cum, r)

# Orient such that the galactic position points in the same direction:
def alignment(s, cen):
  pos_G = -(s['rg'] + cen/1e3) * 1e3
  h = np.linalg.norm(pos_G[[0,1]])
  theta_G = np.sign(pos_G[0]) * np.arcsin(pos_G[1] / h)
  if np.sign(pos_G[0]) == 1: theta_G += np.pi
  Rz = R_z(theta_G)
  s['pos'] = np.dot(s['pos'], Rz)
  return

# Create a relief map:
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interpn
def relief(pos, point_size, width):
  N_bins = 101
  bins = np.linspace(-width/2., width/2., N_bins)
  count, x_e, y_e = np.histogram2d(pos[:,0], pos[:,1], bins=bins)
  count /= (x_e[1] - x_e[0]) * (y_e[1] - y_e[0]) # Convert to count per pc
  count = gaussian_filter(count, sigma=0.5) # Add some smoothing

  Z = interpn(((x_e[1:] + x_e[:-1])/2., (y_e[1:]+y_e[:-1])/2.), count, \
              np.vstack([pos[:,0],pos[:,1]]).T, method="linear", bounds_error=False)
  idx = Z.argsort()

  x, y, Z, point_size = pos[:,0][idx], pos[:,1][idx], np.log10(Z[idx]), point_size[idx]
  Z[Z != Z] = np.nanmin(Z)
  return x, y, Z, point_size

# Bolometric correction for luminosities:
#http://articles.adsabs.harvard.edu/pdf/1998JRASC..92...36R
def BC(T):
  C6 = -8.499
  C7 = 13.421
  C8 = -8.131
  C9 = -3.901
  C10 = -0.438
  T[np.isinf(T)] = 0.
  return C6*(T-4)**4 + C7*(T-4)**3 + C8*(T-4)**2 + C9*(T-4) + C10

def Lbol_Lv(lum, Teff, Mbolsol=4.74, Mvsol=4.83):
  Mbol = (Mbolsol - 2.5 * np.log10(lum)) - BC(Teff)
  return 10**(0.4 * (Mvsol - Mbol))
#=======================================================================
