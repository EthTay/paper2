from config import *

import numpy as np
import tangos
import pynbody
import gc
import importlib

import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
plt.ion()
import time

import arrows

# Load EDGE1 python version.

# Load the appropriate GC IC:
#------------------------------------------------------------
host_sim = 'Halo383_Massive'
GC_intID = 204

# Parse GC properties:
data = load_data()
GC_intIDs = np.array([data[host_sim][i]['Internal ID'] for i in list(data[host_sim].keys())])
GC_ID = np.where(GC_intIDs == GC_intID)[0][0]

GC_IDs = data[host_sim][GC_ID]['IDs']
GC_pos = np.array(data[host_sim][GC_ID]['Galacto-centred position'])
GC_vel = np.array(data[host_sim][GC_ID]['Galacto-centred velocity'])
GC_hlr = data[host_sim][GC_ID]['3D half-mass radius']
GC_mass = data[host_sim][GC_ID]['Stellar Mass'].sum()
GC_Z = data[host_sim][GC_ID]['Median Fe/H']
GC_birthtime = data[host_sim][GC_ID]['Median birthtime']

output = data[host_sim][GC_ID]['Output Number']
halo = data[host_sim][GC_ID]['Tangos Halo ID']
centre = data[host_sim][GC_ID]['Host Uncentered Position']
# Is halo=16 even right? It is never in the main progenitor lineup!

print('Loaded GC data.')
#------------------------------------------------------------

# Load chosen simulation snapshot:
#------------------------------------------------------------
EDGE_path = '/vol/ph/astro_data/shared/etaylor/CHIMERA/'
#EDGE_path = '/vol/ph/astro_data/shared/morkney/EDGE/'

tangos.core.init_db(EDGE_path + host_sim.split('_')[0] + '.db')
session = tangos.core.get_default_session()

# Load simulation snapshot:
h = tangos.get_halo(host_sim + '/output_%05d' % output + '/halo_%i' % (halo+1))
s = pynbody.load(EDGE_path + host_sim + '/output_%05d' % output, maxlevel=1)
s.physical_units()
print('Loaded simulation data.')

# AHF catalogue:
halos = s.halos()
print('Loaded halo data.')
#------------------------------------------------------------

# Find the positions and velocities of all nearby substructure and/or haloes:
#------------------------------------------------------------
host_ID = data[host_sim][GC_ID]['AHF Host ID']
if halos[host_ID].properties['hostHalo'] == -1:
  pass
else:
  host_ID = halos[host_ID].properties['hostHalo']
host_ID += 1

host_pos = np.array([halos[host_ID].properties[i] for i in ['Xc', 'Yc', 'Zc']])
host_vel = np.array([halos[host_ID].properties[i] for i in ['VXc', 'VYc', 'VZc']])
host_mass = halos[host_ID].properties['mass']

children = halos[host_ID].properties['children']
child_pos = np.empty([len(children), 3])
child_vel = np.empty([len(children), 3])
child_mass = np.empty(len(children))
for i, child in enumerate(children):
  child_pos[i] = [halos[child].properties[j] for j in ['Xc', 'Yc', 'Zc']]
  child_vel[i] = [halos[child].properties[j] for j in ['VXc', 'VYc', 'VZc']]
  child_mass[i] = halos[child].properties['mass']

# Centring:
child_pos *= float(s.properties['boxsize']) / s.properties['a']
host_pos *= float(s.properties['boxsize']) / s.properties['a']
child_pos -= host_pos
child_vel -= host_vel

print('Found child properties.')
#------------------------------------------------------------

# 3D plot:
#------------------------------------------------------------
# Normalise the mass between 0-1:
def NormalizeData(data):
  return (data - np.min(data)) / (np.max(data) - np.min(data))

fs = 12
ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')

mass_norm = NormalizeData(np.log10(child_mass))
size = (mass_norm * 20) + 5
colour = cm.viridis_r(np.round(mass_norm * 255).astype('int'))

# Scatter plot:
v_length_factor = 0.025 #0.00005
ax.scatter(*[0,0,0], c='k', s=25, marker='X')
ax.scatter(*child_pos.T, c=colour, s=size, alpha=1)
for i in range(len(children)):
  #v = v_length_factor * child_vel[i] / np.linalg.norm(child_vel[i])
  v = v_length_factor*child_vel[i]
  #ax.plot(*np.transpose([child_pos[i], child_pos[i]+v_norm]), c='r', lw=0.5)

  a = arrows.Arrow3D([child_pos[i][0], child_pos[i][0]+v[0]],
                     [child_pos[i][1], child_pos[i][1]+v[1]],
                     [child_pos[i][2], child_pos[i][2]+v[2]],
                     lw=1, arrowstyle="-|>", color=colour[i], mutation_scale=5)
  ax.add_artist(a)
#------------------------------------------------------------

# Find the mass-averaged velocity for all objects:
#------------------------------------------------------------
mean_vel = np.average(child_vel, weights=child_mass, axis=0)
v = -(v_length_factor * mean_vel)

a = arrows.Arrow3D([0, v[0]],
                   [0, v[1]],
                   [0, v[2]],
                   lw=1.5, arrowstyle="-|>", color="k", mutation_scale=5)
ax.add_artist(a)
#------------------------------------------------------------

# Misc:
#------------------------------------------------------------
ax.set_box_aspect([1,1,1])
ax.tick_params(which='both', axis='both', labelsize=fs-2)
ax.set_xlabel('x [kpc]', fontsize=fs)
ax.set_ylabel('y [kpc]', fontsize=fs)
ax.set_zlabel('z [kpc]', fontsize=fs)

print(r'Relative velocity of host: %.2f,%.2f,%.2f km/s' % (mean_vel[0], mean_vel[1], mean_vel[2]))
#------------------------------------------------------------

# Find the GC stellar IDs:
#------------------------------------------------------------
loc = np.in1d(s.s['iord'], GC_IDs)

# Find the corrected positions and velocities:
pos = s.s['pos'][loc] - host_pos
vel = s.s['vel'][loc] - host_vel
mass = s.s['mass'][loc]
mean_pos = np.average(pos, weights=mass, axis=0)
mean_vel = np.average(vel, weights=mass, axis=0)
#------------------------------------------------------------

fs = 12
ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')

ax.scatter(*s.s['pos'][~loc].T, s=1, c='k')
ax.plot(*s.s['pos'][loc].T, 'm.', zorder=1000, alpha=1)
ax.plot(*host_pos, 'rx', zorder=1000)
ax.set_box_aspect([1,1,1])
ax.tick_params(which='both', axis='both', labelsize=fs-2)
ax.set_xlabel('x [kpc]', fontsize=fs)
ax.set_ylabel('y [kpc]', fontsize=fs)
ax.set_zlabel('z [kpc]', fontsize=fs)

