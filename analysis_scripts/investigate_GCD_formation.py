from config import *

import numpy as np
import tangos
import pynbody
pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
import gc, os, sys
import pickle
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import default_setup
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
#plt.ion()

import rotation_functions as rf

sim_name = sys.argv[1]
GCD_ID = int(sys.argv[2]) # 0,1,4,7NO,   0,1,   0

path='/vol/ph/astro_data/shared/morkney2/GC_paper/Paper1_files/'
EDGE_df = pd.read_csv("%s/Data/complete_2.csv"%(path))
DMC_df = pd.read_csv("%s/Data/DMC_birth_data_2.csv"%(path))

import json
with open("/vol/ph/astro_data/shared/etaylor/cmaps/EDGE_cmap.json", "r") as f:
  EDGE_cmap = LinearSegmentedColormap("/vol/ph/astro_data/shared/etaylor/cmaps/EDGE_cmap.json", json.load(f))
  EDGE_cmap.set_over('w')
  EDGE_cmap.set_bad('w')

metals = {'metal':{}, 'o_metal':{}, 'h':{}}
metals['metal']['mass'] = 9.2732796e-26  # kg
metals['metal']['solar'] = 7.50
metals['h']['mass'] = 1.6737236e-27  # kg
metals['h']['solar'] = 12.0
metals['o_metal']['mass'] = 2.6566962e-26  # kg
metals['o_metal']['solar'] = 8.69
mass_fraction_helium_primordial = 0.24
mass_fraction_metal_primordial = 0.0
mass_fraction_hydrogen = 1 - mass_fraction_metal_primordial - mass_fraction_helium_primordial

# Load DMC data from Ethan's file:
#--------------------------------------------------------------
def load_group_data(sim):
    path = "../../../GC_paper/Paper1_files/Data/%s.starcluster_data" % sim
    loaded_data = [[], []]
    with open(path, "r") as f:
        total_groups = int(f.readline())
        for i in range(total_groups):
            loaded_data[0].append(int(f.readline()))
            num_part = int(f.readline())
            particle_data = np.zeros((num_part, 2), dtype=int)
            for j in range(num_part):
                read_buffer = f.readline().split()
                particle_data[j, 0] = int(read_buffer[0])
                particle_data[j, 1] = int(read_buffer[1])
            loaded_data[1].append(particle_data)
    return loaded_data

def get_snap_info(sim,internal_index):
  snap_data = load_group_data(sim)
  output = snap_data[0][internal_index]
  star_IDs = snap_data[1][internal_index][snap_data[1][internal_index][:, 1] == 4][:, 0]
  DM_IDs = snap_data[1][internal_index][snap_data[1][internal_index][:, 1] == 1][:, 0]
  return output, star_IDs, DM_IDs

output, GCD_IDs_stars, GCD_IDs_dm = get_snap_info(sim_name, GCD_ID)
output_fixed = int(EDGE_df['Output Number'][np.where((EDGE_df['Internal ID']==GCD_ID) & (EDGE_df['Simulation']==sim_name))[0]])
if output!=output_fixed:
  print('OUTPUT NEEDS UPDATING!')
  output = output_fixed
if len(GCD_IDs_dm) < 300:
  print(1/0)
#--------------------------------------------------------------

# Load tangos for this object:
#--------------------------------------------------------------
sim_type = 'CHIMERA' if '383' in sim_name else 'EDGE'
EDGE_path = EDGE_path[sim_type]
TANGOS_path = TANGOS_path[sim_type]
EDGE_output = '%05d' % output
EDGE_halo = 1 # ???
tangos.core.init_db(TANGOS_path + sim_name.split('_')[0] + '.db')
session = tangos.core.get_default_session()

# Find the halo, then find the previous N snapshots leading to this point:
h = tangos.get_halo(('/').join([sim_name, 'output_%s' % EDGE_output, 'halo_%i' % EDGE_halo]))

N = 5
paths = h.next.next.calculate_for_progenitors('path()', nmax=N)[0]
#--------------------------------------------------------------

def find_data(path, sim_name, GCD_ID, GCD_IDs_dm, GCD_IDs_stars):
  '''
  1. Load the raw simulation data.
  2. Identify the GCD DM particles.
  3. Rotate the simulation snapshot to align with these particles.
  4. Create a rho-weighted map of the gas metals.
  5. Plot both the gas metals and the gas density. Include flow lines if possible.
  '''

  #--------------------------------------------------------------
  h = tangos.get_halo(path)
  output = int(h.timestep.extension.split('_')[1])
  if os.path.isfile('./Data/%s_GCD%i_snap%i_gas_density_image.npy' % (sim_name, GCD_ID, output)):
    print('Already exists...')
    return
  else:

    print('>    Output %s' % output)

    # Load full simulation data, full maxlevel:
    s = pynbody.load(EDGE_path + path.split('/')[0] +'/'+ path.split('/')[1])
    s.physical_units()
    s.properties['boxsize'] *= 1e10
    s.g['pos']; s.d['pos']; s.s['pos']
    s.g['mass']; s.d['mass']; s.s['mass']

    try:
      s.s['age'] = pynbody.analysis.ramses_util.get_tform(s)
    except:
      pass

    # Centre:
    GCD_cut_dm = np.in1d(s.d['iord'], GCD_IDs_dm)
    r_center = pynbody.analysis.halo.shrink_sphere_center(s.d[GCD_cut_dm], shrink_factor=0.95)
    s['pos'] -= r_center
    v_center = pynbody.analysis.halo.center_of_mass_velocity(s.d[s.d['r'] <= 0.1])
    s['vel'] -= v_center

    # Calculate metals:
    if len(s.s):
      s.s['Fe/H'] = np.log10((s.s['metal']/metals['metal']['mass']) / (mass_fraction_hydrogen/metals['h']['mass'])) \
                    - (metals['metal']['solar'] - metals['h']['solar'])
    else:
      pass

    # Find the rho-weighted gas densities:
    #width = np.linalg.norm(pos_GCD) * 2.
    width = 5. # [kpc]
    gas_slice = np.abs(s.g['z']) <= width/2.
    dm_slice = np.abs(s.dm['z']) <= width/2.
    img_dm = pynbody.plot.image(s.d[dm_slice], units='Msol kpc^-2', width='%s kpc' % width, ret_im=True, noplot=True)
    img_rho = pynbody.plot.image(s.g[gas_slice], units='Msol kpc^-2', width='%s kpc' % width, ret_im=True, noplot=True)
    img_met = pynbody.plot.image(s.g[gas_slice], qty='metal', width='%s kpc' % width, av_z='rho', ret_im=True, noplot=True)
    img_met = np.log10((img_met/metals['metal']['mass']) / (mass_fraction_hydrogen/metals['h']['mass'])) \
                       - (metals['metal']['solar'] - metals['h']['solar'])

    # Find the star points too:
    if len(s.s):
      star_slice = (np.abs(s.s['z']) <= width/2.) &\
                   (np.abs(s.s['x']) <= width/2.) &\
                   (np.abs(s.s['y']) <= width/2.)

    # Save to file:
    np.save('./Data/%s_GCD%i_snap%i_gas_dm_image.npy' % (sim_name, GCD_ID, output), img_dm)
    np.save('./Data/%s_GCD%i_snap%i_gas_density_image.npy' % (sim_name, GCD_ID, output), img_rho)
    np.save('./Data/%s_GCD%i_snap%i_gas_metal_image.npy' % (sim_name, GCD_ID, output), img_met)
    print('>    Produced images and saved to file.')

    filename = './Data/GCDplots_dict.pk1'
    if not os.path.isfile(filename):
      props = {}
    else:
      with open(filename, 'rb') as file:
        props = pickle.load(file)
    if sim_name not in list(props.keys()):
      props[sim_name] = {}
    if GCD_ID not in list(props[sim_name].keys()):
      props[sim_name][GCD_ID] = {}
    if output not in list(props[sim_name][GCD_ID].keys()):
      props[sim_name][GCD_ID][output] = {}

    if len(s.s):
      props[sim_name][GCD_ID][output]['pos'] = s.s['pos'][star_slice]
      props[sim_name][GCD_ID][output]['Fe/H'] = s.s['Fe/H'][star_slice]
      GCD_cut_stars = np.in1d(s.s['iord'], GCD_IDs_stars)
      props[sim_name][GCD_ID][output]['GCD_stars'] = GCD_cut_stars[star_slice]
    props[sim_name][GCD_ID][output]['GCD_pos'] = r_center
    props[sim_name][GCD_ID][output]['GCD_vel'] = v_center
    props[sim_name][GCD_ID][output]['time'] = s.properties["time"].in_units("Myr")
    with open(filename, 'wb') as file:
      pickle.dump(props, file)
  #--------------------------------------------------------------

  del s
  del h
  gc.collect()

  return

for path in paths:
  find_data(path, sim_name, GCD_ID, GCD_IDs_dm, GCD_IDs_stars)

def plot_data(sim_name, GCD_ID):

  # Load the saved data:
  #--------------------------------------------------------------
  filename = './Data/GCDplots_dict.pk1'
  with open(filename, 'rb') as file:
    props = pickle.load(file)

  width = 5. # [kpc]
  extent = np.array([-1,1,-1,1]) * width/2.
  #--------------------------------------------------------------

  outputs = np.sort([int(path.split('output_')[1][:5]) for path in paths])
  img_dm = ['']*len(outputs)
  img_rho = ['']*len(outputs)
  img_met = ['']*len(outputs)
  for i, output in enumerate(outputs):
    img_dm[i] = np.load('./Data/%s_GCD%i_snap%i_gas_dm_image.npy' % (sim_name, GCD_ID, output))
    img_rho[i] = np.load('./Data/%s_GCD%i_snap%i_gas_density_image.npy' % (sim_name, GCD_ID, output))
    img_met[i] = np.load('./Data/%s_GCD%i_snap%i_gas_metal_image.npy' % (sim_name, GCD_ID, output))

  # Find best colour ranges:
  clim_dm = np.array([np.nanpercentile(np.ravel(j), [1,99.99]) for j in img_dm])
  clim_dm = [np.min(clim_dm[:,0]), np.max(clim_dm[:,1])]
  clim_rho = np.array([np.nanpercentile(np.ravel(j), [1,99.99]) for j in img_rho])
  clim_rho = [np.min(clim_rho[:,0]), np.max(clim_rho[:,1])]
  clim_met = np.array([np.nanpercentile(np.ravel(j), [1,99]) for j in img_met])
  clim_met = [np.min(clim_met[:,0]), np.max(clim_met[:,1])]
  clim_met[0] = max(clim_met[0], -5)

  for i, output in enumerate(outputs):

    print('>    Output %i' % output)

    try:
      GCD = props[sim_name][GCD_ID][output]['GCD_stars']
      pos = props[sim_name][GCD_ID][output]['pos']
      has_stars = True
    except:
      has_stars = False
      pass
    GCD_pos = props[sim_name][GCD_ID][output]['GCD_pos']
    GCD_vel = props[sim_name][GCD_ID][output]['GCD_vel']
    #time = props[sim_name][GCD_ID][output]['time']
    tangos_outputs = [i.extension for i in tangos.get_simulation(sim_name).timesteps]
    tangos_outputs = np.array([i.extension for i in tangos.get_simulation(sim_name).timesteps])
    tangos_timestep = np.where(tangos_outputs == 'output_%05d' % output)[0][0]
    time = tangos.get_simulation(sim_name).timesteps[tangos_timestep].time_gyr * 1e3

    # Add to plot:
    #--------------------------------------------------------------
    extent = np.array([-1,1,-1,1]) * 5./2.

    img_dm_plot = ax[0,i].imshow(img_dm[i], origin='lower', extent=extent, cmap=EDGE_cmap, norm=LogNorm())
    img_dm_plot.set_clim(*clim_dm)
    img_rho_plot = ax[1,i].imshow(img_rho[i], origin='lower', extent=extent, cmap=cm.YlGnBu, norm=LogNorm())
    img_rho_plot.set_clim(*clim_rho)
    img_met_plot = ax[2,i].imshow(img_met[i], origin='lower', extent=extent, cmap=cm.gist_stern)
    img_met_plot.set_clim(*clim_met)

    for j in range(3):
      if has_stars:
        ax[j,i].scatter(pos[~GCD,0], pos[~GCD,1], color='k', s=0.5, edgecolors='none', alpha=0.3)
        ax[j,i].scatter(pos[GCD,0], pos[GCD,1], color='red', s=0.5, edgecolors='none', alpha=1.0)

      ax[j,i].set_xlim(*extent[[0,1]])
      ax[j,i].set_ylim(*extent[[2,3]])
      ax[j,i].set_facecolor('black')
      ax[j,i].set_xticks([])
      ax[j,i].set_yticks([])
      ax[j,i].set_aspect('auto')

    # Time label:
    string = r'$t=%.1f\,\rm{Myr}$' % time
    ax[0,i].text(0.975, 0.975, string, va='top', ha='right', fontsize=fontsize-2, color='w', transform=ax[0,i].transAxes)

    ax[0,i].set_title('Output_%05d' % output, fontsize=fontsize)

  # Distance bar labels:
  ruler = int('%.0f' % float('%.1g' % (width/5)))
  corner1 = width/2. - (0.1*width/2.) - ruler
  corner2 = 0.9*width/2.
  corner1 = 0.1*width/2. - width/2.
  cap = 0.025 * width/2.
  for lw, color, order, capstyle in zip([1.5], ['w'], [100], ['butt']):
    _, _, caps = ax[0,0].errorbar([corner1, corner1+ruler], np.ones(2)*corner2, yerr=np.ones(2)*cap, \
                                  color=color, linewidth=lw, ecolor=color, elinewidth=lw, zorder=order)
    caps[0].set_capstyle(capstyle)
  ax[0,0].text(corner1 + ruler/2., corner2 - 0.025*width/2.,  r'$%.0f\,$kpc' % ruler, \
               va='top', ha='center', color='w', fontsize=fontsize-2)

  clabels = [r'$\rho_{\rm DM}$ [$\rm{M}_{\odot}\,\rm{kpc}^{-2}$]',
             r'$\rho_{\rm gas}$ [$\rm{M}_{\odot}\,\rm{kpc}^{-2}$]',
             r'[Fe/H]']
  for i, (clabel, img) in enumerate(zip(clabels, [img_dm_plot, img_rho_plot, img_met_plot])):
    pad = 0.05
    l, b, w, h = ax[i,-1].get_position().bounds
    cax = fig.add_axes([l+w+w*pad, b+h*pad, w*pad, h-2*h*pad])
    cbar = plt.colorbar(img, cax=cax, extend='both')
    cbar.set_label(clabel, fontsize=fontsize)

  N_zero = np.isinf(props[sim_name][GCD_ID][output]['Fe/H']).sum()
  fig.suptitle(sim_name + ' ' + 'GCD_%i' % GCD_ID + r', $N$(zero metal)=%i' % N_zero, y=0.95)

  return
  #--------------------------------------------------------------

# Initialise figure instance:
#---------------------------------------------------------------------------
fontsize = 10
lw = 1.5
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['axes.linewidth'] = lw
plt.rcParams['lines.linewidth'] = lw
plt.rcParams['lines.markersize'] = 3
plt.rcParams['xtick.labelsize'] = fontsize - 2
plt.rcParams['ytick.labelsize'] = fontsize - 2
plt.rcParams['xtick.major.width'] = lw
plt.rcParams['xtick.minor.width'] = lw * (2/3.)
plt.rcParams['ytick.major.width'] = lw
plt.rcParams['ytick.minor.width'] = lw * (2/3.)
plt.rcParams['legend.fontsize'] = fontsize - 2

row_gap = 0.05
fig, ax = plt.subplots(figsize=(3*(N+1), 3*3), nrows=3, ncols=N+1, gridspec_kw={'hspace':0.0, 'wspace':row_gap})
#---------------------------------------------------------------------------

# Build figure:
#---------------------------------------------------------------------------
for path in paths:
  find_data(path, sim_name, GCD_ID, GCD_IDs_dm, GCD_IDs_stars)

plot_data(sim_name, GCD_ID)

plt.savefig('./Data/%s_GCD%i.pdf' % (sim_name, GCD_ID), bbox_inches='tight')
#---------------------------------------------------------------------------
