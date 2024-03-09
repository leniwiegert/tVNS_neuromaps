# AUTHOR: Lena Wiegert
# Spatial Nulls on parcellated brain maps

import os
import numpy as np
import nibabel as nib
from nilearn import image as nli
from netneurotools import datasets as nntdata
from neuromaps.parcellate import Parcellater
from neuromaps.images import dlabel_to_gifti


# STEPS:
# 1) Transformation from volumetric (MNI152) to surface (fsaverage) space
# 2) Parcellation with Schaefer Atlas
# 3) Spatial correlations
# 4) Spatial nulls

#-------- LOAD AND PREP DATA --------#

# Define universal data directory
data_directory = '/home/leni/Documents/Master/data/'

img = nib.load(os.path.join(data_directory, '4D_rs_fCONF_del_taVNS_sham.nii'))

# Set numpy to print only 2 decimal digits for neatness
np.set_printoptions(precision=2, suppress=True)
# The array proxy allows us to create the image object without immediately loading all the array data from disk
nib.is_proxy(img.dataobj)


# Create mean image
mean_img = nli.mean_img(img)

# Replace non-finite values with a gray matter mask
gray_matter_mask_file = os.path.join(data_directory, 'out_GM_p_0_15.nii')
gray_matter_mask = nib.load(gray_matter_mask_file)

# Resample gray matter mask to match the resolution of mean_img
gray_matter_mask_resampled = nli.resample_to_img(gray_matter_mask, mean_img)

# Ensure both masks have the same shape
if not np.all(gray_matter_mask_resampled.shape == mean_img.shape):
    raise ValueError('Shape of input volume is incompatible.')

# Create a new mask by keeping only non-NaN values in both masks
mean_img_gm_data = np.where(np.isnan(mean_img.get_fdata()), gray_matter_mask_resampled.get_fdata(), mean_img.get_fdata())

# Create a new image with the new mask
mean_img_gm = nib.Nifti1Image(mean_img_gm_data.astype(np.float32), mean_img.affine)

# Make sure the affines are the same
mean_img_gm = nli.resample_to_img(mean_img_gm, img)

# Extract data from Nifti1Image
mean_img_data = mean_img_gm.get_fdata()

'''
#-------- TRANSFORMATION --------#

schaefer = nntdata.fetch_schaefer2018('fslr32k')['400Parcels7Networks']

parc = Parcellater(dlabel_to_gifti(schaefer), 'fsLR')

mean_img_parc = parc.fit_transform(mean_img_gm, 'fsLR')

# nilearn 0.10.3
# netneurotools 0.2.3
# neuromaps 0.0.5
'''

