from config import *

import numpy as np
import pynbody
import tangos
import GC_functions as func
import sys
import os
import pickle

from lmfit import Parameters, Model

import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

#### Currently this script is being reformatted for the new suite versions. Check carefully before using! ####
#### Also check that the bash pipeline scripts are sensible. ####
#### Currently the Mssive +2 output shift is being accounted for, but this might change. ####
#### Also, I am currently treating Full as the reference suite, but that might change too. ####

# Enter EDGE1 to terminal to load pynbody modules

# Simulation choices:
#--------------------------------------------------------------------------
binary_fractions = (0, 0.95)
binary_fraction = binary_fractions[0]
#--------------------------------------------------------------------------

# Parameters:
#--------------------------------------------------------------------------
host_sim = sys.argv[1]
GC_ID = int(sys.argv[2])
plot_fit = False
save_results = True
#--------------------------------------------------------------------------

# Load the GC properties from file:
#--------------------------------------------------------------------------
# Load Ethan's property dict:
data = load_data('recentred')
orig_data = load_data('old')

# Parse GC properties:
GC_pos = np.array(data[host_sim][GC_ID]['Galacto-centred position'])
GC_vel = np.array(data[host_sim][GC_ID]['Galacto-centred velocity'])
GC_hlr = orig_data[host_sim][GC_ID]['3D half-mass radius']
GC_mass = orig_data[host_sim][GC_ID]['Stellar Mass'].sum()
GC_Z = data[host_sim][GC_ID]['Median Fe/H']
GC_birthtime = data[host_sim][GC_ID]['Median birthtime']

# Parse particle properties:
GC_masses = data[host_sim][GC_ID]['Stellar Mass']
GC_metals = data[host_sim][GC_ID]['Fe/H Values']
GC_births = data[host_sim][GC_ID]['Birth Times']

# Parse simulation properties:
if 'Massive' in host_sim:
  EDGE_output = 'output_%05d' % (data[host_sim][GC_ID]['Output Number'] + 2)
else:
  EDGE_output = 'output_%05d' % data[host_sim][GC_ID]['Output Number']
EDGE_sim_name = host_sim
EDGE_halo = data[host_sim][GC_ID]['Tangos Halo ID'] + 1
internal_ID = data[host_sim][GC_ID]['Internal ID'] # Non-exclusive
#--------------------------------------------------------------------------

# Load the simulation snapshot:
#--------------------------------------------------------------------------
sim_type = 'CHIMERA' if '383' in EDGE_sim_name else 'EDGE'
EDGE_path = EDGE_path[sim_type]
TANGOS_path = TANGOS_path[sim_type]
tangos.core.init_db(TANGOS_path + EDGE_sim_name.split('_')[0] + '.db')
session = tangos.core.get_default_session()
h = tangos.get_halo(('/').join([EDGE_sim_name, EDGE_output, 'halo_%i' % EDGE_halo]))
GC_time = h.calculate('t()')
if np.linalg.norm(GC_pos)/1e3 > 40.:
  print('The GC position is beyond the R200 radius. Do not simulate.')
  sys.exit(0)

# Get EDGE simulation density, averaged over nearest 3 simulation snapshots:
r_range = (0.02, 10)
fit_range = (0.035, 3)
if 'CHIMERA' in sim_type:
  data1 = np.loadtxt(path+'/scripts/files/CHIMERA_massive/rho_%s.txt' % h.previous.timestep.extension, unpack=True)
  data2 = np.loadtxt(path+'/scripts/files/CHIMERA_massive/rho_%s.txt' % h.timestep.extension, unpack=True)
  data3 = np.loadtxt(path+'/scripts/files/CHIMERA_massive/rho_%s.txt' % h.next.timestep.extension, unpack=True)
  EDGE_r = np.mean([data1[0], data2[0], data3[0]], axis=0)
  EDGE_rho = np.mean([data1[1], data2[1], data3[1]], axis=0)
else:
  EDGE_r, EDGE_rho = h.previous.calculate_for_descendants('rbins_profile', 'dm_density_profile', nmax=2)
  r = np.logspace(*np.log10(r_range), 100)
  for i in range(3):
    EDGE_r[i], EDGE_rho[i] = func.rebin(EDGE_r[i], EDGE_rho[i], r)
  EDGE_r = EDGE_r[0]
  EDGE_rho = np.mean(EDGE_rho, axis=0)

if np.linalg.norm(GC_pos)/1e3 > fit_range[1]:
  print('GC position is greater that fit_max, %.2f kpc.' % fit_range[1])
elif np.linalg.norm(GC_pos)/1e3 < fit_range[0]:
  print('GC position is less that fit_min, %.2f kpc.' % fit_range[0])
fit_range_arr = (EDGE_r > fit_range[0]) & (EDGE_r <= fit_range[1])

print(f'>    Loaded EDGE density profile for {EDGE_sim_name}, {EDGE_output}.')
#--------------------------------------------------------------------------

# Reconstruct the potential at this time:
#--------------------------------------------------------------------------
print('>    Finding the background potential at this time...')
def interp(param, alpha):
  return alpha*param[0] + (1-alpha)*param[1]

filename = path+'/scripts/files/host_profiles_dict.pk1'
with open(filename, 'rb') as file:
  props = pickle.load(file)

sim = '_'.join([EDGE_sim_name, suite])
time = props[sim.replace('_compact', '').replace('_Raw', '_Full')]['time']
rs = props[sim.replace('_compact', '').replace('_Raw', '_Full')]['rs']
gamma = props[sim.replace('_compact', '').replace('_Raw', '_Full')]['gamma']
Mg = props[sim.replace('_compact', '').replace('_Raw', '_Full')]['Mg']
i = np.where((GC_birthtime > time[:-1]*1e3) & (GC_birthtime <= time[1:]*1e3))[0][0]

alpha = (GC_birthtime - time[i]*1e3) / (time[i+1]*1e3 - time[i]*1e3)
rs = interp(rs[[i,i+1]], alpha)
gamma = interp(gamma[[i,i+1]], alpha)
#--------------------------------------------------------------------------

# Plot the result and compare:
#--------------------------------------------------------------------------
if plot_fit:
  fs = 14
  fig, ax = plt.subplots(figsize=(6, 6))

  ax.loglog(EDGE_r, EDGE_rho, 'k', lw=2, label='%s, %s' % (EDGE_sim_name, EDGE_output))
  ax.axvline(np.linalg.norm(GC_pos/1e3), c='k', lw=1)

  label = r'Fit %i: ' % 0 + \
          r'$r_{\rm s}=%.2f$, ' % (rs) + \
          r'$M_{\rm g}=%s\,$M$_{\odot}$, ' % func.latex_float(Mg) + \
          r'$\gamma=%.2g$' % gamma
  ax.loglog(EDGE_r, func.Dehnen_profile(EDGE_r, np.log10(rs), np.log10(Mg), gamma), ls='--', lw=1, label=label)

  ax.axvspan(EDGE_r.min(), EDGE_r[fit_range_arr].min(), facecolor='k', alpha=0.1)
  ax.axvspan(EDGE_r[fit_range_arr].max(), EDGE_r.max(), facecolor='k', alpha=0.1)

  ax.set_xlim(*EDGE_r[[0,-1]])

  ax.set_ylabel(r'$\rho_\mathrm{tot}$ [M$_{\odot}$ kpc$^{-3}$]', fontsize=fs)
  ax.set_xlabel('Radius [kpc]', fontsize=fs)

  ax.tick_params(axis='both', which='both', labelsize=fs-2)

  ax.legend(loc='lower left', fontsize=fs-4)
#--------------------------------------------------------------------------

# Correct output back again:
if 'Massive' in host_sim:
  EDGE_output = 'output_%05d' % (int(EDGE_output.split('_')[1]) - 2)
  h = tangos.get_halo(('/').join([EDGE_sim_name, EDGE_output, 'halo_%i' % EDGE_halo]))

# Trace the orbit backwards until the time of birth:
#--------------------------------------------------------------------------
print('>    Tracing the GC orbit back to birth...')
import agama

# Set up units:
agama.setUnits(mass=1, length=1, velocity=1)
timeunit = pynbody.units.s * pynbody.units.kpc / pynbody.units.km
Gyr_to_timeunit = pynbody.array.SimArray(1, units='Gyr').in_units(timeunit)

Dehnen_potential = agama.Potential(type='Dehnen', mass=Mg, scaleRadius=rs, gamma=gamma)

v_circ = func.Dehnen_vcirc(r=np.linalg.norm(GC_pos/1e3), rs=rs, Mg=Mg, gamma=gamma, G=agama.G)
v_circ_astro = pynbody.array.SimArray(v_circ, units='km s^-1').in_units('kpc Gyr^-1')
orbital_distance = 2 * np.pi * np.linalg.norm(GC_pos/1e3)
period = orbital_distance / v_circ_astro
birthtime = (GC_time - GC_birthtime/1e3) / Gyr_to_timeunit # Gyr

if suite == 'Full':
  # Calculate orbits in the Full potential:
  total_time = 1 / Gyr_to_timeunit # Gyr
  time_steps = 1000 # Myr precision
  phase = np.append(GC_pos/1e3, GC_vel * -1)
  orbits = agama.orbit(ic=phase, potential=Dehnen_potential, time=total_time, trajsize=time_steps)

  # Retrieve the position and velocity at the time of birth:
  birthindex = np.abs(orbits[0] - birthtime).argmin()
  GC_pos_birth = orbits[1][birthindex,[0,1,2]]
  GC_vel_birth = orbits[1][birthindex,[3,4,5]] * -1
else:
  Full_file = path + f'Nbody6_sims/Full_files/{EDGE_sim_name}_{EDGE_output}_{internal_ID}.txt'
  with open(Full_file, 'r') as f:
    for line in f:
      pass
  line = np.array(line.strip('\n').split(' ')).astype('float')
  GC_pos_birth = line[[0,1,2]]
  GC_vel_birth = line[[3,4,5]]

# Find orbital peri and apo-centre in the fitted potential starting from birth location:
Rperi, Rapo = Dehnen_potential.Rperiapo(np.append(GC_pos_birth, GC_vel_birth))

if plot_fit:
  ax.text(Rperi, 0.99, r'$R_{\rm peri}$', fontsize=fs-2, color='r', rotation=90, ha='right', va='top', transform=ax.get_xaxis_transform())
  ax.text(Rapo, 0.99, r'$R_{\rm apo}$', fontsize=fs-2, color='r', rotation=90, ha='left', va='top', transform=ax.get_xaxis_transform())
  ax.axvline(Rperi, c='r', lw=0.5, zorder=-100)
  ax.axvline(Rapo, c='r', lw=0.5, zorder=-100)
  ax.axvspan(Rperi, Rapo, facecolor='r', alpha=0.1, zorder=-100)
  ax.axvline(np.linalg.norm(GC_pos_birth), c='r', lw=1, zorder=100)

print('>    Integrated orbit backwards by %.2f Gyr with Agama.' % birthtime)
#--------------------------------------------------------------------------

# Scale the mass according to the stellar mass evolution:
#--------------------------------------------------------------------------
print('>    Deciding the GC initial mass...')
GC_time = h.calculate('t()')

# Don't correct the mass if this is the Raw suite:
if 'Raw' not in suite:
  # If the Full suite has already been built, reuse the corrected mass:
  if os.path.isfile(f'../../Nbody6_sims/Full_files/{EDGE_sim_name}_{EDGE_output}_{internal_ID}.txt'):
    with open(f'../../Nbody6_sims/Full_files/{EDGE_sim_name}_{EDGE_output}_{internal_ID}.txt') as f:
      f.readline()
      GC_mass = float(f.readline())
    print('Copied modified initial mass from Full suite.')

  # Reverse-integrate the stellar mass using Eric's functions:
  elif suite=='Full':
    import stellar_devolution_functions as StellarDevolution
    param = StellarDevolution.Parameters('EDGE1')

    # Integrate each particle individually:
    dt = 0.1 # [Myr]
    GC_mass = 0
    for i in range(len(GC_masses)):
      smass_array = np.zeros(100)
      # Randomly evolve the star 100 times and take the average:
      for j in range(100):
        stars = StellarDevolution.Stars(npartmax=1)
        stars.add_stars(GC_masses[i], 10**GC_metals[i], GC_time*1e3-GC_births[i])
        for k in range(int(np.round((GC_time*1e3 - GC_births[i]) / dt))):
          stars.evolve(dt, param)
        smass_array[j] = stars.mass[0]
      GC_mass += np.mean(smass_array)
    print('Modified initial mass.')
  else:
    print('Problem! There is no modified mass available from the reference suite.')
    print(1/0)
#--------------------------------------------------------------------------

# Scale the half-light radius if necessary:
#--------------------------------------------------------------------------
if 'compact' in suite:
  filename = path+'/scripts/files/adjusted_hmr.pk1'
  with open(filename, 'rb') as file:
    hmr_props = pickle.load(file)

  GC_hlr = hmr_props[EDGE_sim_name][GC_ID]

  print('Modified initial size.')
#--------------------------------------------------------------------------

# Save the results to a file:
#--------------------------------------------------------------------------
if save_results:
  if not os.path.isdir(path + f'Nbody6_sims/{suite}_files/'):
    os.mkdir(path + f'Nbody6_sims/{suite}_files/')
  with open(path + f'Nbody6_sims/{suite}_files/{EDGE_sim_name}_{EDGE_output}_{internal_ID}.txt', 'w') as file:
    file.write(sim + '\n')
    file.write('%.8f\n' % GC_mass)
    file.write('%.8f\n' % GC_hlr)
    file.write('%.8f\n' % 10**GC_Z)
    file.write('%.8f\n' % (GC_birthtime/1e3))
    file.write('%.8f\n' % binary_fraction)
    file.write('0.0 0.0 0.0 0.0 0.0 0.0 %.6f \n' % Mg)
    file.write('%.6f %.6f %.6f %.6f %.6f %.6f\n' % (GC_pos_birth[0], GC_pos_birth[1], GC_pos_birth[2], \
                                                    GC_vel_birth[0], GC_vel_birth[1], GC_vel_birth[2]))

  print('>    Parameter file saved to ' + path + f'Nbody6_sims/{suite}_files/{EDGE_sim_name}_{EDGE_output}_{internal_ID}.txt')
else:
  print('>    Parameter file not saved.')
#--------------------------------------------------------------------------
