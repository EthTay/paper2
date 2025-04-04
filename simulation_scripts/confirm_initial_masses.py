from config import *
import pickle
import numpy as np
import glob

path = '/vol/ph/astro_data/shared/morkney2/GC_mock_obs/'
data = load_data('recentred')

# Find the masses given in the dictionary:
def get_dict():
  return np.concatenate([[i + '_output_%05d' % data[i][j]['Output Number'] + '_%i' % data[i][j]['Internal ID'] for j in list(data[i].keys())] for i in list(data.keys())])
ICs = get_dict()
mass_ICs = np.concatenate([[data[i][j]['Stellar Mass'].sum() for j in list(data[i].keys())] for i in list(data.keys())])

# Load up the raw particle data from elsewhere:
