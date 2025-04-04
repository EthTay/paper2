import numpy as np
import h5py
import itertools
import glob
import sys
import gc

import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as path_effects
from labellines import labelLine, labelLines
plt.ion()

import pickle
from scipy.stats import binned_statistic_2d
from scipy.ndimage import median_filter

# Load GC data:
#--------------------------------------------------------------------
suite = 'Full'
with open('../files/GC_data_%s.pk1' % suite, 'rb') as f:
  GC_data = pickle.load(f)
#--------------------------------------------------------------------

def get_prop(d, prop, id=0):
  return np.array([GC_data[i][prop][id] if prop in GC_data[i].keys() else np.nan for i in list(GC_data.keys())])

# Find the simulations that have not finished yet (mainly unbound objects):
sims = np.array([i for i in list(GC_data.keys())])
mass = get_prop(GC_data, 'mass', id=-1)
print(sims[np.where(mass > 500)])

ra = get_prop(GC_data, 'ra')
rp = get_prop(GC_data, 'rp')
print(1/0)
# A plot illustrating the initial orbital eccentricity distributions:
#--------------------------------------------------------------------
fs = 12
fig, ax = plt.subplots(figsize=(6,6))

sims = np.array([i.split('_output')[0] for i in list(GC_data.keys())])
sim_names = np.array(sorted(np.unique(sims)))[[0,3,4,5,2,1]]
colours = ['#DDCC77', '#CC6677', '#882255', '#332288', '#117733', '#88CCEE']
for i, (sim, colour) in enumerate(zip(sim_names, colours)):
  select = sim == sims
  ax.scatter(ra[select], rp[select], s=10, facecolor='None', edgecolor=colour, zorder=100-i, label=sim.replace('_hires', '').replace('_fiducial', ''))
ax.loglog()

ax.legend(loc='upper left', fontsize=fs-2, ncol=2, handletextpad=0.2, columnspacing=1.)

ax.tick_params(which='both', axis='both', labelsize=fs-2)

ax.set_ylabel('Pericentre [kpc]', fontsize=fs)
ax.set_xlabel('Apocentre [kpc]', fontsize=fs)

# Add lines of constant eccentricity:
y_range = ax.get_ylim()
x_range = ax.get_xlim()

x_bins = np.logspace(*np.log10(x_range), 100)
y_limit = x_bins
ax.fill_between(x_bins, y_limit, y_range[1], color='silver')

for ecc in np.linspace(0,1,5):
  ra_fixed_ecc = (x_bins - ecc*x_bins) / (1+ecc)
  ax.plot(x_bins, ra_fixed_ecc, ls='--', color='grey', lw=1, label=r'$e=%.2f$' % ecc)

ax.set_ylim(y_range)
ax.set_xlim(x_range)

for i in range(6):
  ax.collections[i]._label = ""
kwargs = {'align':True, 'alpha':1, 'fontsize':fs-2, 'va':'top', 'ha':'left', 'outline_width':1.5, 'zorder':1000}
labelLines(ax.get_lines(), xvals=(8e1,8e1,8e1,8e1,8e1), **kwargs)

ax.set_title(suite, fontsize=fs)
#--------------------------------------------------------------------

# Survival time:
#--------------------------------------------------------------------
t_ini = get_prop(GC_data, 't', id=0)
t_fin = get_prop(GC_data, 't', id=-1)
t_fin[t_fin > 13.82] = 13.82

fs = 12
fig, ax = plt.subplots(figsize=(6,6), nrows=2, gridspec_kw={'hspace':0.0})

t_ini_list = []
t_fin_list = []
for i, (sim, colour) in enumerate(zip(sim_names, colours)):
  select = sim == sims
  t_ini_list.append(t_ini[select])
  t_fin_list.append(t_fin[select])

labels = [sim.replace('_hires', '').replace('_fiducial', '') for sim in sim_names]
t_bins = np.arange(0, 14.5, 0.5)
ax[0].hist(t_fin_list, t_bins, histtype='bar', stacked=True, color=colours)
ax[1].hist(t_ini_list, t_bins, histtype='bar', stacked=True, color=colours, label=labels)

ax[0].set_xlim(0, 13.82)
ax[1].set_xlim(0, 13.82)
ax[1].invert_yaxis()

ax[1].set_xlabel('Time [Gyr]', fontsize=fs)
ax[0].set_ylabel(r'$N$ (Final)', fontsize=fs)
ax[1].set_ylabel(r'$N$ (Initial)', fontsize=fs)
ax[0].tick_params(which='both', axis='both', labelsize=fs-2)
ax[1].tick_params(which='both', axis='both', labelsize=fs-2)

ax[0].set_yscale('log')
ax[1].set_yscale('log')

ax[1].legend(loc='lower right', fontsize=fs-2, ncol=2, handletextpad=0.2, columnspacing=1.)

ax[0].set_title(suite, fontsize=fs)
#--------------------------------------------------------------------
