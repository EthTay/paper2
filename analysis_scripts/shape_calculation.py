from config import *
import numpy as np
from read_out3 import read_nbody6
import GC_functions as func

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

from glob import glob
import pickle
import time
import os

import shape as sh
import rotation_functions as rf

suite = 'Full'
overwrite = True
lum_max = 100

def get_shape(task, stars_per_bin=1000, limits=[25.5, 30]):

  global s

  R = task[0]
  i = task[1]

  a_ = np.zeros(3)
  b_ = np.zeros(3)
  R_ = [''] * 3
  s_i = s[i].copy()

  # Rotate the positions into place:
  s_i['pos'] = np.dot(s_i['pos'], R)
  s_i['r2'] = np.linalg.norm(s_i['pos'][:,:2], axis=1)

  # Calculate the shape in regular bins with equal particles:
  r_max = np.percentile(s_i['r2'][s_i['nbound']], 99) * 3
  N_in_r_max = np.sum(s_i['r2'] <= r_max)
  N_bins = max(int(np.round(N_in_r_max / stars_per_bin)), 5)
  shape = sh.shape(s_i, cut=np.ones_like(s_i['r']).astype('bool'), nbins=N_bins, rmin=0, rmax=r_max, weight='lum', ndim=2)
  a = shape[2][:,0]
  b = shape[2][:,1]

  # Calculate the surface brightness within each bin:
  inner = np.transpose(shape[1][:-1]*(shape[2].T/shape[0]))
  outer = np.transpose(shape[1][1:]*(shape[2].T/shape[0]))
  area_homoeoid = np.pi * ((outer[:,0] * outer[:,1]) - (inner[:,0] * inner[:,1]))
  in_homoeoid = [(sh.Ellipsoid(s_i['pos'][:,:2], inner[j], shape[4][j]) > 1) & \
                 (sh.Ellipsoid(s_i['pos'][:,:2], outer[j],  shape[4][j]) < 1) for j in range(N_bins)]

  # Luminosity per pc^2 (equation from https://en.wikipedia.org/wiki/Surface_brightness):
  surface_lum = np.array([s_i['lum'][k].sum() / j for j,k in zip(area_homoeoid, in_homoeoid)])
  surface_mag = 4.83 + 21.572 - 2.5 * np.log10(surface_lum) # [Converts from pc to arcsecs]

  # Cull all the stars that fall outside of the surface brightness cut:
  for j, limit in enumerate(limits):
    surface_index = np.where(surface_mag <= limit)[0]
    if len(surface_index):
      a_b_interp = [np.interp(limit, surface_mag, outer[:,0]), np.interp(limit, surface_mag, outer[:,1])]
      surface_cut = sh.Ellipsoid(s_i['pos'][:,:2], a_b_interp, shape[4][surface_index[-1]]) < 1

      # Calculate the final shape of the GC:
      if np.sum(surface_cut) > 5:
        shape_ = sh.shape(s_i, cut=surface_cut, nbins=1, rmin=0, weight='lum', ndim=2)
        a_[j+1] = shape_[2][0]
        b_[j+1] = shape_[2][1]
        R_ [j+1] = shape_[4]
      else:
        a_[j+1] = np.nan
        b_[j+1] = np.nan
        R_[j+1] = np.nan
    else:
     a_[j+1] = np.nan
     b_[j+1] = np.nan
     R_[j+1] = np.nan

  # Also calculate shape for just the bound particles:
  if np.sum(s_i['nbound']) > 5:
    shape_ = sh.shape(s_i, cut=s_i['nbound'], nbins=1, rmin=0, weight='lum', ndim=2)
    a_[0] = shape_[2][0]
    b_[0] = shape_[2][1]
    R_[0] = shape_[4]
  else:
    a_[0] = np.nan
    b_[0] = np.nan
    R_[0] = np.nan

  s_i['pos'] = np.dot(s_i['pos'], R.T)

  return a_, b_, R_

# Prepare pseudo-observation angles:
#------------------------------------------------------------
# Find a series of N_obs_angles rotation matrices:
N_obs_angles = 100
x, y, z = sh.points_to_sphere(N_obs_angles * 2)

# Filter out all the observation angles which are symmetrical:
hemisphere = np.where(z>=0)[0]
x, y, z = x[hemisphere], y[hemisphere], z[hemisphere]

# Find the corresponding rotation matrices:
R_obs = [sh.rotation_matrix_from_vectors([l,n,m], [0,0,1]) for l,n,m in zip(x,y,z)]
#------------------------------------------------------------

# Load the property dictionary:
#------------------------------------------------------------
filename = path + f'scripts/files/GC_shapedata_{suite}.pk1'
if os.path.isfile(filename):
  with open(filename, 'rb') as f:
    GC_data = pickle.load(f)
else:
  GC_data = {}

filename2 = path + f'scripts/files/GC_data_{suite}.pk1'
if os.path.isfile(filename2):
  with open(filename2, 'rb') as f2:
    GC_data2 = pickle.load(f2)
#------------------------------------------------------------

# Load Ethan's property dict:
#------------------------------------------------------------
data = load_data('recentred')

def get_dict(key):
  return np.concatenate([[data[i][j][key] for j in list(data[i].keys())] for i in list(data.keys())])

GC_ID = get_dict('Internal ID')
GC_birthtime = get_dict('Median birthtime') # [Myr]
#------------------------------------------------------------

# Prepare multiprocessing:
#------------------------------------------------------------
import multiprocessing
n_threads = 16
#------------------------------------------------------------

# Loop over all simulations:
#------------------------------------------------------------
sims = glob(path + f'Nbody6_sims/{suite}/Halo*')
#sims = ['/vol/ph/astro_data/shared/morkney2/GC_mock_obs/Nbody6_sims/Full/Halo383_fiducial_early_output_00040_61']
for sim in sims:

  # Management:
  #------------------------------------------------------------
  start = time.time()
  print(sim)
  ID = int(''.join(c for c in sim.split('_')[-1] if c.isdigit()))
  birthtime = GC_birthtime[np.where(GC_ID == ID)[0][0]]

  # Various accounting:
  if sim.split('/')[-1] not in list(GC_data.keys()):
    GC_data[sim.split('/')[-1]] = {}
    print('>    Creating new data entry.')
  elif not len(GC_data[sim.split('/')[-1]]):
    print('>    Creating new data entry.')
  elif (sim.split('/')[-1] in list(GC_data.keys())) & \
     (GC_data[sim.split('/')[-1]]['t'][-1] <= (13.8+birthtime)) & \
     (GC_data2[sim.split('/')[-1]]['mass'][-1] >= 1e3):
    print('>    Updating data entry on %s...' % sim.split('/')[-1])
    pass
  elif GC_data2[sim.split('/')[-1]]['rg'][-1] > 10e3:
    print('>    Unbound entry.')
    continue
  elif (sim.split('/')[-1] in list(GC_data.keys())) & (not overwrite):
    print('>    Data entry already exists.')
    continue

  try:
    s = read_nbody6(sim, df=True)
  except:
    print('>    Data entry has no outputs.')
    continue
  sim = sim.split('/')[-1]
  print('>    %i' % ID, end='')
  #------------------------------------------------------------

  # Initialise property arrays:
  #------------------------------------------------------------
  GC_data[sim] = {}
  GC_properties = ['t', 'SBcut_25.5', 'SBcut_30', 'Nbound_cut']
  GC_data[sim]['t'] = np.ones(len(s)) * np.nan
  for GC_property in GC_properties[1:]:
    GC_data[sim][GC_property] = {}
    GC_data[sim][GC_property]['a'] = np.ones([len(s), N_obs_angles]) * np.nan
    GC_data[sim][GC_property]['b'] = np.ones([len(s), N_obs_angles]) * np.nan
    GC_data[sim][GC_property]['R'] = np.ones([len(s), N_obs_angles]) * np.nan
  #------------------------------------------------------------

  # Centre and trim the data:
  #------------------------------------------------------------
  for i in range(min(len(s), len(GC_data2[sim]['t']))):
    if GC_data[sim]['t'][i] == (birthtime + s[i]['age']) / 1e3:
      continue

    # Get the simulation time:
    GC_data[sim]['t'][i] = (birthtime + s[i]['age']) / 1e3 # Gyr

    # Centre the GC:
    body_noBHs = s[i]['nbound'] & (s[i]['kstara'] != 14)
    if not np.sum(body_noBHs):
      continue
    cen = func.shrink(s[i])
    s[i]['pos'] -= cen
    s[i]['r'] = np.linalg.norm(s[i]['pos'], axis=1)

    # Trimming:
    lum_filter = (s[i]['lum'] <= lum_max) & (s[i]['lum'] > 1e-10)
    for key in s[i].keys() - ['pos', 'lum', 'nbound', 'r']:
      del s[i][key]
    for key in s[i].keys():
      s[i][key] = s[i][key][lum_filter]

    # Orient the GC in its orbit:
    R_matrix1 = rf.align_orbit(GC_data2[sim]['posg'][i], [0,1,0])
    pos_GC2 = np.dot(R_matrix1, GC_data2[sim]['posg'][i]).T

    # Orient the frame on the GC velocity:
    R_matrix2 = rf.align_orbit(pos_GC2, np.dot(R_matrix1, GC_data2[sim]['velg'][i]))

    # Final matrix has GC orbit in x-y plane, travelling in +x and located in +y directions:
    R_matrix = np.dot(R_matrix2, R_matrix1)
    s[i]['pos'] = np.dot(s[i]['pos'], R_matrix.T)
  #------------------------------------------------------------

  # Shape calculations in parallel:
  #------------------------------------------------------------
  pool = multiprocessing.Pool(n_threads)
  for i in range(len(s)):
    print('>    %i, %.2fs' % (i, time.time()-start))
    tasks = np.array([R_obs, np.ones(len(R_obs)).astype('int')*i]).T
    shape = pool.map(get_shape, tasks)

    for j, R in enumerate(R_obs):
      for k, cut in enumerate(['Nbound_cut', 'SBcut_25.5', 'SBcut_30']):
        GC_data[sim][cut]['a'][i][j] = shape[j][0][k]
        GC_data[sim][cut]['b'][i][j] = shape[j][1][k]
        GC_data[sim][cut]['R'][i][j] = sh.almnt(shape[j][2][k])
  pool.close()
  #------------------------------------------------------------

  print(', %.2fs' % (time.time() - start))

  print('Saving...')
  with open(filename, 'wb') as f:
    pickle.dump(GC_data, f)
#------------------------------------------------------------
