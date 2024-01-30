
from neuromaps import datasets, images, nulls

# This function returns nulls (generated null distribution, where each column represents a unique null map)

# Call annotations
#hesse2017_MNI152 = datasets.fetch_annotation(source='hesse2017')
vol_1 = f'/Users/leni/Documents/Master/Data/volume_1.nii'

# Calculate spatial nulls
#nulls = nulls.burt2018(hesse2017_MNI152, atlas='MNI152', density='3mm', n_perm=100, seed=1234)
nulls = nulls.moran(vol_1, atlas='MNI152', density='3mm', n_perm=100, seed=1234)

print(nulls.shape)


