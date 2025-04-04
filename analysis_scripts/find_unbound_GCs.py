import numpy as np

import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as path_effects
plt.ion()

import pickle

# Load GC data:
#--------------------------------------------------------------------
suite = 'Full'
with open('../files/GC_data_%s.pk1' % suite, 'rb') as f:
  GC_data = pickle.load(f)
#--------------------------------------------------------------------

# For each GC, find the maximum orbital radius ever:
#--------------------------------------------------------------------
sims = list(GC_data.keys())
R_max = np.zeros(len(sims))
for i, sim in enumerate(sims):
  try:
    R_max[i] = GC_data[sim]['rg'][-1]
  except:
    pass
#--------------------------------------------------------------------

# Weirdos:
#--------------------------------------------------------------------
weirdos = np.array(sims)[R_max >= 10 * 1e3] # [kpc]
for weirdo in weirdos:
 print(weirdo)
#--------------------------------------------------------------------
