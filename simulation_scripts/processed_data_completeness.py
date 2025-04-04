from config import *
import pickle
import numpy as np
import glob

suites = ['Full', 'Full_compact', 'Fantasy_cores', 'Fantasy_cores_compact']
path = '/vol/ph/astro_data/shared/morkney2/GC_mock_obs/'
data = load_data('recentred')

def get_dict():
  return np.concatenate([[i + '_output_%05d' % data[i][j]['Output Number'] + '_%i' % data[i][j]['Internal ID'] for j in list(data[i].keys())] for i in list(data.keys())])

for suite in suites:

  print(suite)

  with open(path + f'scripts/files/GC_data_{suite}.pk1', 'rb') as file:
    props = pickle.load(file)

  # Find database entries:
  entries = np.array(list(props.keys()))
  #entries = entries[['Massive' not in entry for entry in entries]]

  # Find available simulations:
  sims = np.array(glob.glob(path + '/Nbody6_sims/' + suite + '/*'))
  #sims = sims[['Massive' not in sim for sim in sims]]
  sims = sims[['Halo' in sim for sim in sims]]
  sims = np.array([sim.split('/')[-1] for sim in sims])
  if 'Fantasy' in suite:
    sims = sims[['383' not in sim for sim in sims]]

  # Find Ethan's ICs:
  ICs = get_dict()
  if 'Fantasy' in suite:
    ICs = ICs[['383' not in IC for IC in ICs]]

  # Cross-match:
  matched = np.in1d(sims, entries)
  print('    There are %i sims.' % len(sims))
  print('    The following simulations have no database entry:')
  print(sims[~matched])

  matched = np.in1d(ICs, entries)
  print('    There are %i ICs.' % len(ICs))
  print('    The following ICs have no database entry:')
  print(ICs[~matched])

  nomatch = np.in1d(entries, ICs)
  print('    The following simulations should be removed from the database:')
  print(entries[~nomatch])

  '''
  for entry in entries[~nomatch]:
    del props[entry]
  with open(path + f'scripts/files/GC_data_{suite}.pk1', 'wb') as file:
    pickle.dump(props, file)
  '''

  print()
