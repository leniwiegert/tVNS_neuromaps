'''
@author: Lena Wiegert

This code transforms MNI152 data to fsLR space and parcellates it with the Schaefer atlas.
It is based on this workshop: https://www.youtube.com/watch?v=pc8zMMTLxmA
'''

import os
import numpy as np
import nibabel as nib
from nilearn import image as nli
import matplotlib.pyplot as plt
from tqdm import trange
from nilearn import datasets, surface
from netneurotools import datasets as nntdata
from netneurotools.datasets import fetch_schaefer2018
from neuromaps import transforms, images, nulls, parcellate
from neuromaps import datasets
from neuromaps.nulls import alexander_bloch
from neuromaps.parcellate import Parcellater
from neuromaps.datasets import fetch_annotation
from neuromaps.images import (construct_shape_gii, load_data, annot_to_gifti,
                              relabel_gifti, dlabel_to_gifti)


#-- Debugging --#
import os
print(os.environ['PATH'])

#-- Set matplotlib properties --#
# s. workshop at 52:00


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

# Save the masked data
nib.save(mean_img_gm, os.path.join(data_directory, 'mean_img_gm.nii'))

# Path to masked data
path_mean_img_gm = os.path.join(data_directory, 'mean_img_gm.nii')


#-------- TRANSFORMATION + PARCELLATION --------#

# Fetching annotation
anno = fetch_annotation(source='ding2010')

# Workshop example without parcellation (fsaverage):
# The Workshop shows how to fetch all annotations in one line (1:00:00)
# Transforming the image to fsaverage (1:04:00)
# He works with: ('neurosynth', 'cogpc1', 'MNI152', '2mm')
# Using alexander.bloch on the transformed fsaverage image (1:06:00)
# Saving nulls for both hemispheres separately (1:08:00)
# Comparing of transformed image to anno that originally is in fsaverage (1:10:00)
# Resampling + spatial calculations + Plotting

# Workshop example with parcellation:
# Defines plot function for surface plot (1:15:00)

# Import the parcellation maps (Schaefer) in all spaces

# fsaverage41k
parcels_fsav_41k = fetch_schaefer2018('fsaverage6')['400Parcels7Networks']
parcels_fsav_41k = annot_to_gifti((parcels_fsav_41k))
parcels_fsav_41k = relabel_gifti(parcels_fsav_41k)

# fsLR32k
parcels_fslr_32k = fetch_schaefer2018('fslr32k')['400Parcels7Networks']
parcels_fslr_32k = dlabel_to_gifti(parcels_fslr_32k)
parcels_fslr_32k = relabel_gifti(parcels_fslr_32k)

# mni152 (different than in the workshop!)
#parcels_mni152 = fetch_schaefer2018('mni152')['100Parcels7Networks']

# Create parcellaters for all spaces
parc_fsav = Parcellater(parcels_fsav_41k, 'fsaverage', resampling_target='parcellation')
parc_fsLR = Parcellater(parcels_fslr_32k, 'fslr', resampling_target=None)
#parc_mni152 = Parcellater(parcels_mni152, 'mni152', resampling_target='parcellation')

# Transform my image to fsLR (not in the workshop!)
mean_img_fslr = transforms.mni152_to_fslr(mean_img_gm, '32k', method='nearest')
print(mean_img_fslr)
# The output is a tuple of Gifti Images

# Parcellate my image
mean_img_fslr_parc = parc_fsLR.fit_transform(mean_img_fslr, 'fsLR')
# The output is an array

# Generate nulls
nulls = alexander_bloch(mean_img_fslr_parc, atlas='fsLR', density='32k', parcellation=parcels_fslr_32k)
print(nulls)
print(len(nulls))
# Should be 400

plt.hist(nulls, bins=30)
plt.xlabel('Null Values')
plt.ylabel('Frequency')
plt.title('Null Distribution for the Mean Image in fsLR (400 ROIs)')
plt.show()


# Compare to other fsLR map for spatial correlations












