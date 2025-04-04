import pynbody
import numpy as np
import pandas as pd
import gc

import json

def load_group_data(sim, path='/vol/ph/astro_data/shared/morkney2/GC_paper/Paper1_files/Data/'):
    path += "/%s.starcluster_data" % sim
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

  snap = snap_data[0][internal_index]
  star_IDs = snap_data[1][internal_index][snap_data[1][internal_index][:, 1] == 4][:, 0]
  DM_IDs = snap_data[1][internal_index][snap_data[1][internal_index][:, 1] == 1][:, 0]

  return sim,snap,star_IDs,DM_IDs

def mutual(halo_ids,check_ids):
  tf=np.in1d(halo_ids,check_ids)
  return (len(tf[tf])**2)/(len(halo_ids)*len(check_ids))
