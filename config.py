import numpy as np

# Select simulation suite:
#---------------------------------------------------------------------
# SUITES:
# __________________________________________________________________________________________________________________________________________________
#| (0) DM         | (1) Full          | (2) Fantasy_cores   | (3) DM_compact | (4) Full_compact | (5) Fantasy_cores_compact | (6) Raw               |
#| Fit to DM only | Fit to all matter | Forced gamma=0 fits | Adjusted hmr   | Adjusted hmr     | Adjusted hmr              | Original mass and hmr |
#| Incomplete     | Incomplete        | Incomplete          | Incomplete     | Incomplete       | Incomplete                | Incomplete            |
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
suites = ['DM', 'Full', 'Fantasy_cores', 'DM_compact', 'Full_compact', 'Fantasy_cores_compact', 'Raw']
suite = suites[1]
print(suite)
path = '/vol/ph/astro_data/shared/morkney2/GC_mock_obs/'
#---------------------------------------------------------------------

# Simulation and tangos paths:
#---------------------------------------------------------------------
EDGE_path = {'EDGE':'/vol/ph/astro_data/shared/morkney/EDGE/', \
             'CHIMERA':'/vol/ph/astro_data/shared/etaylor/CHIMERA/'}
TANGOS_path = {'EDGE':'/vol/ph/astro_data/shared/morkney/EDGE/tangos/', \
               'CHIMERA':'/vol/ph/astro_data/shared/etaylor/CHIMERA/'}
#---------------------------------------------------------------------

# Ethan's GC data dictionary:
#---------------------------------------------------------------------
def load_data(version='new'):
  import pickle5 as pickle
  if version=='new':
    with open('/vol/ph/astro_data/shared/etaylor/paper2/data/GC_Nbody.data', 'rb') as handle:
      b = pickle.load(handle)
  elif version=='old':
    with open('/vol/ph/astro_data/shared/etaylor/paper2/data/GC_Nbody_2.data', 'rb') as handle:
      b = pickle.load(handle)
  elif version=='recentred':
     with open("/vol/ph/astro_data/shared/etaylor/paper2/data/GC_Nbody_383_center.data", 'rb') as handle:
      b = pickle.load(handle)
  elif version=='mega':
     with open("/vol/ph/astro_data/shared/etaylor/paper1/data/GC_survivors.data", 'rb') as handle:
      b = pickle.load(handle)
  else:
    print('No version %s' % version)
    return
  return b
#---------------------------------------------------------------------
