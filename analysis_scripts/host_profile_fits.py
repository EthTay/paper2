from config import *

import tangos

import numpy as np
import sys
import os

import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcols
plt.ion()

import GC_functions as func

import pickle

# Load potential fits and GC_data:
#--------------------------------------------------------------------------
filename = path+'/scripts/files/host_profiles_dict.pk1'
if os.path.isfile(filename):
  with open(filename, 'rb') as file:
    props = pickle.load(file)

with open('../files/GC_data_%s.pk1' % suite, 'rb') as f:
  GC_data = pickle.load(f)

def get_prop(d, prop, id=0):
  return np.array([GC_data[i][prop][id] if prop in GC_data[i].keys() else np.nan for i in list(GC_data.keys())])

sims = np.array([i for i in list(GC_data.keys())])
ra = get_prop(GC_data, 'ra')
rp = get_prop(GC_data, 'rp')
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
hosts = ['Halo1459_fiducial_hires', \
         'Halo600_fiducial_hires', \
         'Halo605_fiducial_hires', \
         'Halo624_fiducial_hires', \
         'Halo383_fiducial_early', \
         'Halo383_Massive']
hosts = [hosts[0]]

def NormaliseData(data):
  return (data - np.min(data)) / (np.max(data) - np.min(data))

def interp(param, alpha):
  return alpha*param[0] + (1-alpha)*param[1]

fs = 10
fig, ax = plt.subplots(figsize=(6+0.35*2, 3*len(hosts)), ncols=2, nrows=len(hosts), gridspec_kw={'wspace':0.35, 'hspace':0.3})

N_bins = 200
r_bins = np.logspace(np.log10(0.02), np.log10(10), N_bins)

for i, host in enumerate(hosts):

  print(host)

  profile = host + '_' + suite

  sim_type = 'CHIMERA' if '383' in host else 'EDGE'
  EDGE_path_ = EDGE_path[sim_type]
  TANGOS_path_ = TANGOS_path[sim_type]
  tangos.core.init_db(TANGOS_path_ + host.split('_')[0] + '.db')
  session = tangos.core.get_default_session()
  output = tangos.get_simulation(host).timesteps[-1].extension
  h = tangos.get_halo(host + '/' + output + '/' + 'halo_1')

  if sim_type=='EDGE':
    # Find the density evolution through time:
    EDGE_t, EDGE_rho, EDGE_r = h.calculate_for_progenitors('t()', 'dm_density_profile+gas_density_profile+star_density_profile', 'rbins_profile')

    # Rebin the density:
    for j in range(len(EDGE_t)):
      EDGE_r[j], EDGE_rho[j] = func.rebin(EDGE_r[j], EDGE_rho[j], r_bins)
  else:
    # Load profile dictionary:
    filename = path+'/scripts/files/CHIMERA_properties_dict.pk1'
    if os.path.isfile(filename):
      with open(filename, 'rb') as file:
        props2 = pickle.load(file)
    else:
      sys.exit(0)
    # Unpack:
    outputs = list(props2[host].keys())
    EDGE_t = np.array([props2[host][output]['t'] for output in outputs])
    EDGE_r = np.array([props2[host][output]['r'] for output in outputs])
    EDGE_rho = np.array([props2[host][output]['DM_rho'] +
                         props2[host][output]['gas_rho'] +
                         props2[host][output]['star_rho'] for output in outputs])

  # Add the fits through time:
  rho_at_r = [np.sqrt(0.001), 0.1, np.sqrt(0.1), 1.0]
  mean_rhos1 = np.zeros([len(EDGE_t), len(rho_at_r)])
  mean_rhos2 = np.zeros([len(EDGE_t), len(rho_at_r)])
  diff = np.zeros([len(EDGE_t), len(EDGE_r[0])])
  ts = props[profile]['time']
  for j, t in enumerate(EDGE_t):
    mean_rhos1[j] = [np.interp(k, EDGE_r[j], EDGE_rho[j]) for k in rho_at_r]
    loc = np.where(ts-t <= 0)[0][-1]
    alpha = (t - ts[loc]) / (ts[loc+1] - ts[loc])

    Mg = props[profile]['Mg']
    rs = interp([props[profile]['rs'][loc], props[profile]['rs'][loc+1]], alpha)
    gamma = interp([props[profile]['gamma'][loc], props[profile]['gamma'][loc+1]], alpha)
    fit_rho = func.Dehnen_profile(EDGE_r[j], np.log10(rs), np.log10(Mg), gamma)

    mean_rhos2[j] = [np.interp(k, EDGE_r[j], fit_rho) for k in rho_at_r]

    diff[j] = (EDGE_rho[j] / fit_rho) - 1.

  anom_norm = mcols.SymLogNorm(
        linthresh=0.1,
        linscale=0.001,
        vmin=-2,
        vmax=2)

  X, Y = np.meshgrid(EDGE_r[0], EDGE_t)
  ax[1].pcolormesh(X, Y, diff, cmap=cm.coolwarm_r, norm=anom_norm)
  ax[1].set_xscale('log')
  ax[1].set_aspect('auto')

  '''
  colours = ['r', 'g', 'b', 'c']
  for j in range(len(rho_at_r)):
    ax[1].semilogy(EDGE_t, mean_rhos1[:,j], color=colours[j], alpha=0.3)
    ax[1].semilogy(EDGE_t, mean_rhos2[:,j], color=colours[j])
  '''

  N_t = 100
  t_bins = np.linspace(EDGE_t.min(), EDGE_t.max(), N_t)
  #t_bins = np.logspace(np.log10(0.3), np.log10(13.8), N_t)
  colors = cm.rainbow(NormaliseData(t_bins))
  #colors = cm.rainbow(NormaliseData(t_bins**(1/2.)))
  # LOOKOUT! The colourbar doesn't match these values...

  for j, (t, color) in enumerate(zip(t_bins, colors)):
    loc = np.where(ts-t <= 0)[0][-1]
    alpha = (t - ts[loc]) / (ts[loc+1] - ts[loc])

    Mg = props[profile]['Mg']
    rs = interp([props[profile]['rs'][loc], props[profile]['rs'][loc+1]], alpha)
    gamma = interp([props[profile]['gamma'][loc], props[profile]['gamma'][loc+1]], alpha)
    rho = func.Dehnen_profile(r_bins, np.log10(rs), np.log10(Mg), gamma)

    ax[0].loglog(r_bins, rho, lw=0.5, c=color, zorder=j)

  #for j in range(len(rho_at_r)):
  #  ax[0].axvline(rho_at_r[j], c=colours[j], ls='--')

  # Add GC starting positions:
  '''
  select = [host in sim for sim in sims]
  size = 0.225
  l, b, w, hh = ax[0].get_position().bounds
  hax_x = fig.add_axes([l, b+hh, w, hh*size], zorder=-1)

  hist = np.histogram(ra[select]/1e3, bins=r_bins)
  hax_x.fill_between(x=r_bins[1:], y1=0, y2=hist[0], color='g', step='pre', lw=1, facecolor='None', clip_on=False, label=r'$r_{\rm apo, ini}$')
  hist = np.histogram(rp[select]/1e3, bins=r_bins)
  hax_x.fill_between(x=r_bins[1:], y1=0, y2=hist[0], color='m', step='pre', lw=1, facecolor='None', clip_on=False, label=r'$r_{\rm peri, ini}$')
  hax_x.set_ylim(ymin=0)
  hax_x.set_xlim(*r_bins[[0,-1]])
  hax_x.set_xscale('log')
  hax_x.tick_params(axis='both', which='both', labelsize=fs-2)
  hax_x.set_ylabel(r'$N$', fontsize=fs)
  hax_x.set_xticklabels([])
  hax_x.legend(loc='upper right', fontsize=fs-2)
  '''

  ax[0].tick_params(axis='both', which='both', labelsize=fs-2)
  ax[1].tick_params(axis='both', which='both', labelsize=fs-2)

  ax[0].set_ylabel(r'Density [M$_{\odot}\,{\rm kpc}^{-3}$]', fontsize=fs)
  #ax[1].set_ylabel(r'Time [Gyr]', fontsize=fs)
  ax[0].set_xlabel('Radius [kpc]', fontsize=fs)
  ax[1].set_xlabel('Radius [kpc]', fontsize=fs)

  ax[0].set_xlim(*r_bins[[0,-1]])
  ax[1].set_xlim(*r_bins[[0,-1]])

ax[0].set_aspect(np.diff(np.log10(ax[0].get_xlim())) / np.diff(np.log10(ax[0].get_ylim())))
ax[1].set_aspect(np.diff(np.log10(ax[1].get_xlim())) / np.diff(ax[1].get_ylim()))

# Add a small cmap:
norm = mpl.colors.Normalize(vmin=ax[1].get_ylim()[0], vmax=ax[1].get_ylim()[1])
sm = cm.ScalarMappable(norm=norm, cmap=cm.rainbow)
sm.set_array([])
width = 0.05
l, b, w, h = ax[1].get_position().bounds
cax = fig.add_axes([l-width*w, b, width*w, h])
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label('Time [Gyr]', fontsize=fs)
cax.tick_params(axis='both', which='both', labelsize=fs-2)
cax.yaxis.set_label_position('left')
cax.yaxis.set_ticks_position('left')
ax[1].set_yticklabels([])

norm = mpl.colors.Normalize(vmin=-2, vmax=2)
sm = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm_r)
sm.set_array([])
pad = 0.05
l, b, w, h = ax[1].get_position().bounds
cax = fig.add_axes([l, b+h+h*pad, w, h*pad])
cbar = plt.colorbar(sm, cax=cax, orientation="horizontal", extend='max')
cbar.set_label(r'$\log_{10}$ residual (percent)', fontsize=fs-2)
cax.tick_params(axis='both', which='both', labelsize=fs-2)
cax.xaxis.set_label_position('top')
cax.xaxis.set_ticks_position('top')

#ax[0].text(0.05, 0.05, host.replace('_',''), fontsize=fs, ha='left', va='bottom', transform=ax[0].transAxes)
#--------------------------------------------------------------------------

plt.savefig('../images/hostfit_%s.pdf' % hosts[0], bbox_inches='tight')
