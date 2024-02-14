
import os
import numpy as np
import nibabel as nib
from abagen import annot_to_gifti, relabel_gifti
from nilearn import image as nli
from netneurotools import datasets as nntdata
from netneurotools.datasets import fetch_schaefer2018
from neuromaps.parcellate import Parcellater
from neuromaps.images import dlabel_to_gifti
from nilearn.surface import Surface
from neuromaps import transforms, images, nulls



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


#-------- TRANSFORMATION --------#

# Transform MNI152 to fsLR (surface based data) with the density we want it in
# Nearest-neighbors  interpolation recommended for parcellation data
mean_img_fsav = transforms.mni152_to_fsaverage(mean_img_gm, '41k', method='nearest')
print(mean_img_fsav)
# The output is a tuple of Gifti Images

# Compare shapes of volumetric vs. fsLR/fsaverage data
print(mean_img_data.shape)
# (97, 115, 97)
print(images.load_data(mean_img_fsav).shape)
# (327684,), depending on the density you're in
mean_img_fsav_lh, mean_img_fsav_rh = mean_img_fsav
print(mean_img_fsav_lh.agg_data().shape)
# (163842,)


# Check attributes of the left hemisphere image
print("Left Hemisphere Gifti Image:")
for array in mean_img_fsav_lh.darrays:
    print(array.data)

# Check attributes of the right hemisphere image
print("Right Hemisphere Gifti Image:")
for array in mean_img_fsav_rh.darrays:
    print(array.data)

# this part makes total sense, for fsLR and for fsaverage
# BUT: i need to make sure that mean_img_fsav correctly represents both hemispheres!
# it does :)


#-------- PARCELLATION --------#

from neuromaps.images import load_gifti

# Parcellation with netneurotools
schaefer_fsav_41k = fetch_schaefer2018('fsaverage6')['400Parcels7Networks']
print(schaefer_fsav_41k)

# Obtain the file paths from the Surface object
lh_annotation_file = schaefer_fsav_41k.lh
rh_annotation_file = schaefer_fsav_41k.rh

# Convert annotation files to GIfTI images
lh_gifti_file = annot_to_gifti(lh_annotation_file)
rh_gifti_file = annot_to_gifti(rh_annotation_file)

# Load data from GIfTI files
lh_gifti_data = load_gifti(lh_gifti_file)
rh_gifti_data = load_gifti(rh_gifti_file)

# Create parcellater for left and right hemisphere separately
parc_fsav_lh = Parcellater(lh_gifti_file, 'fsaverage', resampling_target='parcellation', hemi='L')
parc_fsav_rh = Parcellater(rh_gifti_file, 'fsaverage', resampling_target='parcellation', hemi='R')

# Parcellate data for LH and RH separately
mean_img_fsav_lh_parc = parc_fsav_lh.fit_transform(mean_img_fsav_lh, 'fsaverage', hemi='L')
mean_img_fsav_rh_parc = parc_fsav_rh.fit_transform(mean_img_fsav_rh, 'fsaverage', hemi='R')
print(mean_img_fsav_lh_parc.shape)
print(mean_img_fsav_rh_parc.shape)



#--------- SPATIAL NULL MODELS --------#

# Fetch annotation
#hesse2017 = datasets.fetch_annotation(source='hesse2017')

# Calculate nulls with hungarian function
mean_img_parc_nulls = nulls.hungarian(mean_img_fsav_lh_parc, atlas='fsaverage', density='10k', parcellation=schaefer_fsav_41k, n_perm=10)
print(mean_img_parc_nulls.shape)
# Error: 'str' object has no attribute 'agg_data'

# Alternative:
#rotated = nulls.alexander_bloch(mean_img_fsav_lh_parc, atlas='fsaverage', density='10k',
#                                n_perm=100, seed=1234, parcellation=schaefer_fsav_41k)

#print(rotated.shape)
# Same error

# Parcellate hesse2017 before this next step
#corr, pval = stats.compare_images(mean_img_fsav_lh_parc, hesse2017_parc, nulls=rotated)
#print(f'r = {corr:.3f}, p = {pval:.3f}')


