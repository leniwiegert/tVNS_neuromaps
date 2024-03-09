
import os
import neuromaps.nulls
import numpy as np
import nibabel as nib
from nilearn import image as nli
from neuromaps.parcellate import Parcellater


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

from neuromaps import transforms, images

# Transform MNI152 to fsLR (surface based data) with the density we want it in
mean_img_fsav = transforms.mni152_to_fsaverage(mean_img_gm, '10k')
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

from nilearn.surface import vol_to_surf

from nilearn.datasets import fetch_atlas_schaefer_2018

# Download atlas
schaefer = fetch_atlas_schaefer_2018(n_rois=400)

# File path to the parcellation file
print('Path to annotation file:', schaefer['maps'])

# Load the annotation file
schaefer_path = '/home/leni/nilearn_data/schaefer_2018/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz'
schaefer_img = nib.load(schaefer_path)

# Convert the annotation file to gifti images
schaefer_gii = vol_to_surf(schaefer_img, mean_img_fsav_lh)

# Print information about the output gifti images
print(schaefer_gii)

schaefer['maps'], 'fsaverage', hemi='L')
# Path to parcellation image + space in which schaefer is defined (MNI152 and fsaverage)

# Parcellate non-transformed data
#mean_img_parc = parcellater.fit_transform(mean_img, 'MNI152')
#print(mean_img_parc.shape)
# Should be (1,400)
# You now have the volumetric brain image parcellated in 400 rois

# Parcellate transformed data
mean_img_fsav_parc = parcellater.fit_transform(mean_img_fsav_lh, 'fsaverage', hemi='L')
print(mean_img_fsav_parc.shape)


'''
#--------- SPATIAL NULL MODELS --------#

from neuromaps import nulls, datasets

# Fetch atlas
hesse2017 = datasets.fetch_annotation(source='hesse2017')

# Calculate nulls of transformed data
#sc_rotated = nulls.alexander_bloch(mean_img_fsav_parc, atlas='fslr', density='32k', n_perm=10)
#print(sc_rotated.shape)

# Problem:
# Transformation of parcellated data in surface space needed to use alexander_bloch
# That changes the shape of the data from a tuple to an array (?)
# Tuple needed for null models
# Do I even need alexander.bloch?
# No, alexander.bloch is for unparcellated data!

# Alternative: Hungarian function

#neuromaps.nulls.hungarian(mean_img_parc, atlas='fsaverage', density='10k', parcellation=schaefer['maps'], n_perm=10)

'''



