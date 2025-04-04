from config import *

import numpy as np
import pynbody
import tangos
import sys
import os
import pickle
import glob

import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

# Key data:
# Mass
# Half light radius

# Load the IC data:
data = load_data('recentred')

def get_dict():
  return np.concatenate([[i + '_output_%05d' % data[i][j]['Output Number'] + '_%i' % data[i][j]['Internal ID'] for j in list(data[i].keys())] for i in list(data.keys())])
ICs = get_dict()
IDs = np.concatenate([[data[i][j]['Internal ID'] for j in list(data[i].keys())] for i in list(data.keys())])
hmr_ICs = np.concatenate([[data[i][j]['3D half-mass radius'] for j in list(data[i].keys())] for i in list(data.keys())])
mass_ICs = np.concatenate([[data[i][j]['Stellar Mass'].sum() for j in list(data[i].keys())] for i in list(data.keys())])

suite = 'Full_old'

# Load the data as given by the IC files:
hmr_file = np.zeros(len(mass_ICs))
Full_mass_file = np.zeros(len(mass_ICs))
for n, IC in enumerate(ICs):
  try:
    with open('../../Nbody6_sims/%s_files/%s.txt' % (suite, IC)) as f:
      f.readline()
      Full_mass_file[n] = float(f.readline())
      hmr_file[n] = float(f.readline())
  except:
    continue

Full_mass_file2 = np.zeros(len(mass_ICs))
for n, IC in enumerate(ICs):
  try:
    f = np.genfromtxt('../../Nbody6_sims/%s/%s/fort.10' % (suite, IC))
    Full_mass_file2[n] = np.sum(f[:,0])
  except:
    try:
      f = np.genfromtxt('../../Nbody6_sims/%s/%s/run1/fort.10' % (suite, IC))
      Full_mass_file2[n] = np.sum(f[:,0])
    except:
      continue

suite = 'Full_compact_old'

# Load the data as given by the IC files:
hmr_file = np.zeros(len(mass_ICs))
compact_mass_file = np.zeros(len(mass_ICs))
for n, IC in enumerate(ICs):
  try:
    with open('../../Nbody6_sims/%s_files/%s.txt' % (suite, IC)) as f:
      f.readline()
      compact_mass_file[n] = float(f.readline())
      hmr_file[n] = float(f.readline())
  except:
    continue

compact_mass_file2 = np.zeros(len(mass_ICs))
for n, IC in enumerate(ICs):
  try:
    f = np.genfromtxt('../../Nbody6_sims/%s/%s/fort.10' % (suite, IC))
    compact_mass_file2[n] = np.sum(f[:,0])
  except:
    try:
      f = np.genfromtxt('../../Nbody6_sims/%s/%s/run1/fort.10' % (suite, IC))
      compact_mass_file2[n] = np.sum(f[:,0])
    except:
      continue


plt.plot(np.diff([compact_mass_file, Full_mass_file], axis=0)[0])
plt.plot(np.diff([compact_mass_file2, Full_mass_file2], axis=0)[0])
plt.plot(np.diff([Full_mass_file, Full_mass_file2], axis=0)[0])
plt.plot(np.diff([compact_mass_file, compact_mass_file2], axis=0)[0])

