'''
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import nilearn
from nilearn import plotting
from nilearn import image as nli
from nilearn.regions import connected_regions
from neuromaps.datasets import fetch_annotation
from neuromaps.resampling import resample_images
from nilearn.plotting import plot_roi
from matplotlib.colors import Normalize
from nilearn.image import resample_img
from neuromaps import stats
from neuromaps import datasets, images, nulls

# Change temp directory
#export TMPDIR=/mnt/ghrelin/newtmp/

# This function returns nulls (generated null distribution, where each column represents a unique null map)

img = nib.load('/home/neuromadlab/tVNS_project/data/4D_rs_fCONF_del_taVNS_sham.nii')
mean_img = nli.mean_img(img)

# Replace non-finite values with a gray matter mask
gray_matter_mask_file = '/home/neuromadlab/tVNS_project/data/out_GM_p_0_15.nii'  # Provide the path to your gray matter mask
gray_matter_mask = nib.load(gray_matter_mask_file)

# Choose either mean_img for all volumes or mean_img_vol_1-41 for the desired volume
cmap_mask_img = mean_img

# Resample gray matter mask to match the resolution of cmap_mask_img
gray_matter_mask_resampled = nli.resample_to_img(gray_matter_mask, cmap_mask_img)

# Ensure both masks have the same shape
if not np.all(gray_matter_mask_resampled.shape == cmap_mask_img)
#nulls = nulls.burt2018(hesse2017_MNI152, atlas='MNI152', density='3mm', n_perm=100, seed=1234)
nulls = nulls.moran(combined_mask_img, atlas='MNI152', density='3mm', n_perm=3, seed=1234)

print(nulls.shape)

'''

# TEST: Convert the data from int64 to int32
# Further reduction would most likely lead to a significant loss of precision

import numpy as np
import nibabel as nib
from nilearn import image as nli
from nilearn.image import resample_to_img
from neuromaps import nulls
from tqdm import tqdm  # Import tqdm for progress bar

# Load the 4D rs fCONF del taVNS sham image
img = nib.load('/home/neuromadlab/tVNS_project/data/4D_rs_fCONF_del_taVNS_sham.nii')
mean_img = nli.mean_img(img)

# Provide the path to your gray matter mask
gray_matter_mask_file = '/home/neuromadlab/tVNS_project/data/out_GM_p_0_15.nii'
gray_matter_mask = nib.load(gray_matter_mask_file)

# Choose either mean_img for all volumes or mean_img_vol_1-41 for the desired volume
cmap_mask_img = mean_img

# Resample gray matter mask to match the resolution of cmap_mask_img
gray_matter_mask_resampled = resample_to_img(gray_matter_mask, cmap_mask_img)

# Ensure both masks have the same shape
if not np.all(gray_matter_mask_resampled.shape == cmap_mask_img.shape):
    raise ValueError('Shape of input volume is incompatible.')

# Create a new mask by keeping only non-NaN values in both masks
combined_mask_data = np.where(np.isnan(cmap_mask_img.get_fdata()), gray_matter_mask_resampled.get_fdata(), cmap_mask_img.get_fdata())

# Create a new image with the combined mask
combined_mask_img = nib.Nifti1Image(combined_mask_data.astype(np.float32), cmap_mask_img.affine)

# Calculate spatial nulls with int32 data type
num_permutations = 3
with tqdm(total=num_permutations, desc="Calculating nulls", unit="permutation") as pbar:
    nulls = nulls.moran(combined_mask_img, atlas='MNI152', density='3mm', n_perm=num_permutations, seed=1234, callback=lambda x: pbar.update(1))

print(nulls.shape)


