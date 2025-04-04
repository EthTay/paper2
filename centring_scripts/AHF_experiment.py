from config import *

import tangos
import pynbody
import numpy as np
import gc

import glob
import pandas as pd
import loading_functions as lf

import matplotlib.pyplot as plt
plt.ion()

# Options:
#--------------------------------------------------------------------------
sim_name = 'Halo383_Massive'
tangos_path = '/vol/ph/astro_data/shared/morkney2/GC_mock_obs/scripts/centring_scripts/Halo383_AHF.db'
#--------------------------------------------------------------------------

# Load the simulation database:
#--------------------------------------------------------------------------
tangos.core.init_db(tangos_path)
sim_type = 'CHIMERA'
EDGE_path = EDGE_path[sim_type]
session = tangos.core.get_default_session()
#--------------------------------------------------------------------------

# Grab the main halo that exists at the final output:
#--------------------------------------------------------------------------
tangos.all_simulations() # Only Halo383_Massive
tangos.get_simulation('Halo383_Massive').timesteps # Goes all the way to output 1
h = tangos.get_halo('Halo383_Massive/output_00172/halo_1') # This is the correct object.

# When running with EDGE1:
### AttributeError: module 'tangos.input_handlers.pynbody' has no attribute 'RamsesAHFInputHandler'
# Works OK when running with EDGE2!

h['shrink_center']

# Find the properties through time:
cen, hnum, fnum, t, z, paths = h.calculate_for_progenitors('shrink_center', 'halo_number()', 'finder_id()', 't()', 'z()', 'path()')

# Whoops, there is a disconnection... Oh no!
end_path = paths[-1]
end_h = tangos.get_halo(end_path)
join_h = end_h.previous
cen2, hnum2, fnum2, t2, z2, paths2 = join_h.calculate_for_progenitors('shrink_center', 'halo_number()', 'finder_id()', 't()', 'z()', 'path()')

# Connect up the two disconnected lineages:
cen = np.append(cen, cen2, axis=0)
hnum = np.append(hnum, hnum2)
fnum = np.append(fnum, fnum2)
t = np.append(t, t2)
z = np.append(z, z2)
paths = np.append(paths, paths2)

outputs = np.array([int(path.split('/')[-2].split('output_')[1]) for path in paths])
#--------------------------------------------------------------------------

# Try looking at the movement of the centre:
#--------------------------------------------------------------------------
a = 1 / (1+z)
pos = cen / np.vstack(a)

plt.plot(outputs, pos)

# There appear to be a few visible disconnections in the position of the halo. Strange.
#--------------------------------------------------------------------------

# Look at the GC data:
#--------------------------------------------------------------------------
path = '/vol/ph/astro_data/shared/morkney2/GC_paper/Paper1_files/Data'
used_sims = np.array(['Halo383_Massive'])
dwarfs_df = pd.read_csv("%s/refresh_dwarfs_updated.csv" % (path))
tf = np.in1d(dwarfs_df['Simulation'], used_sims)

# No file for 383_Massive available here, maybe Ethan has one...:
snap_data = lf.load_group_data(sim_name, path=path)
#--------------------------------------------------------------------------

# Try checking outputs 26,27,28 in sequence and making sure the centring lines up sensibly!
#--------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12,4), nrows=1, ncols=3)
sim_outputs = [41, 42, 43]
for i, sim_output in enumerate(sim_outputs):
  s = pynbody.load(EDGE_path + sim_name +'/output_%05d' % sim_output, maxlevel=1)
  s.physical_units()

  # Take off the centre from the AHF tangos catalogue:
  loc = np.where(outputs == sim_output)
  s.s['pos'] -= cen[loc]
  s.d['pos'] -= cen[loc]

  # Load AHF catalogue and find the corresponding object:
  halos = s.halos()
  halo = halos[int(hnum[loc])]
  # Whoops! All dummy haloes...

  # Double check the positional centre and find the velocity centre too:
  ax[i].plot(s.s['x'], s.s['y'], 'k,')
  ax[i].plot(0, 0, 'rx')

  # Get the CoM velocity:
  #pynbody.analysis.halo.center_of_mass_velocity(s[s['r'] <= 1])
  vel_cen = pynbody.analysis.halo.vel_center(s, cen_size='1 kpc', retcen=True)
  ax[i].text(0.05, 0.95, '%.2f, %.2f, %.2f' % (vel_cen[0], vel_cen[1], vel_cen[2]), ha='left', va='top', transform=ax[i].transAxes)

  ax[i].set_xlim(-20, 20)
  ax[i].set_ylim(-20, 20)

  # Looks like the main "chunk" is being tracked correctly.
  # The vel_cen values are quite high, possibly due to attraction with an incoming major merger.
  # I think the position of the GC needs to come in here. Is it orbiting the infalling object? What?
  # vel_cen stabilises between snapshots when I increase the cen_size argument. The separate inspiralling clumps seem to matter.

  # It seems that the GCs with "unbound" velocities are actually attached to the incoming object rather than the main progenitor.
  # However, if I take the velocity of the incoming object then it is not too crazy...

  # Load the raw data:
  sim, snap, star_IDs, DM_IDs = lf.get_snap_info(sim_name, 204)
  stars = np.in1d(s.s["iord"], star_IDs)
  ax[i].plot(s.s['x'][stars], s.s['y'][stars], 'r.')

  del s
  gc.collect()
#--------------------------------------------------------------------------
