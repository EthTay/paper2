import numpy as np
import pickle5 as pickle

# Open the new and old database files:
with open("/vol/ph/astro_data/shared/etaylor/paper2/data/GC_Nbody_2.data", "rb") as handle:
  a = pickle.load(handle)
with open("/vol/ph/astro_data/shared/etaylor/paper2/data/GC_Nbody.data", "rb") as handle:
  b = pickle.load(handle)

# There are a different number of objects in each file, each with different numbers.
# Therefore, I must carefully compare between internal IDs!:

sim = "Halo383_Massive"
#sim = "Halo605_fiducial_hires"

internal_IDb = []
posb = []
velb = []
outputb = []
metal = []
hlrb = []
for i in b[sim].keys():
  internal_IDb.append(b[sim][i]['Internal ID'])
  posb.append(b[sim][i]['Galacto-centred position'])
  velb.append(b[sim][i]['Galacto-centred velocity'])
  outputb.append(b[sim][i]['Output Number'])
  metal.append(b[sim][i]['Median Fe/H'])
  hlrb.append(b[sim][i]['3D half-mass radius'])
internal_IDa = []
posa = []
vela = []
outputa = []
hlra = []
for i in a[sim].keys():
  internal_IDa.append(a[sim][i]['Internal ID'])
  posa.append(a[sim][i]['Galacto-centred position'])
  vela.append(a[sim][i]['Galacto-centred velocity'])
  outputa.append(a[sim][i]['Output Number'])
  hlra.append(a[sim][i]['3D half-mass radius'])
internal_IDb = np.array(internal_IDb)
internal_IDa = np.array(internal_IDa)
posb = np.array(posb)
posa = np.array(posa)
velb = np.array(velb)
metal = np.array(metal)
vela = np.array(vela)
outputb = np.array(outputb)
outputa = np.array(outputa)
hlrb = np.array(hlrb)
hlra = np.array(hlra)

same_cut = np.in1d(internal_IDa, internal_IDb)
posa = posa[same_cut]
vela = vela[same_cut]
internal_IDa = internal_IDa[same_cut]

the_same_pos = np.all(np.isclose(posa, posb, atol=5), axis=1)
the_same_vel = np.all(np.isclose(vela, velb, atol=5), axis=1)
