import nibabel
import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
from nilearn import image as nli
import arviz
import numpy as np
import nibabel as nib
import seaborn as sns
import pandas as pd
from nilearn import image as nli
from nibabel import gifti as nlg
import matplotlib.pyplot as plt
from netneurotools.datasets import fetch_schaefer2018
from neuromaps import transforms, stats
from neuromaps.resampling import resample_images
from neuromaps.nulls import alexander_bloch
from neuromaps.parcellate import Parcellater
from neuromaps.datasets import fetch_annotation
from neuromaps.images import (relabel_gifti, dlabel_to_gifti)


# Define universal data directory
data_directory = '/home/leni/Documents/Master/data/'

# Paths to the atlas and MNI152 NIFTI files
#atlas_path = '/home/leni/Tian2020MSA/Tian2020MSA/3T/Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz'
atlas_path = '/home/leni/Tian2020MSA/Tian2020MSA/3T/Cortex-Subcortex/Schaefer2018_400Parcels_17Networks_order_Tian_Subcortex_S4.dlabel.nii.gz'
mni152_path = os.path.join(data_directory, 'combined_mask.nii.gz')


def load_atlas(atlas_path):
    """Load the NIFTI atlas file."""
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    atlas_affine = atlas_img.affine
    return atlas_data, atlas_affine

# Load the atlas
atlas_data, _ = load_atlas(atlas_path)

# Get unique regions
unique_regions = np.unique(atlas_data)

# Print number of unique regions and their values
print(f"Number of unique regions: {len(unique_regions)}")
print(f"Unique regions: {unique_regions}")


def load_mni152(mni152_path):
    """Load the MNI152 NIFTI file."""
    mni152_img = nib.load(os.path.join(data_directory, 'combined_mask.nii.gz'))
    mni152_data = mni152_img.get_fdata()
    mni152_affine = mni152_img.affine
    return mni152_data, mni152_affine


def parcellate_mni152_with_atlas(mni152_data, atlas_data):
    """Parcellate the MNI152 data using the atlas."""
    unique_regions = np.unique(atlas_data)
    parcellated_data = np.zeros_like(mni152_data)

    for region in unique_regions:
        if region == 0:  # Skip background
            continue
        mask = atlas_data == region
        parcellated_data[mask] = region

    print(f"MNI152 data shape: {mni152_data.shape}")
    print(f"Atlas data shape: {atlas_data.shape}")

    return parcellated_data


#### SUBCORTICAL MEAN IMAGE PARCELLATION
# Paths to the atlas and MNI152 NIFTI files
atlas_path = '/home/leni/Tian2020MSA/Tian2020MSA/3T/Subcortex-Only/Tian_Subcortex_S2_3T.nii.gz'
mni152_path = os.path.join(data_directory, 'combined_mask.nii.gz')

# Load the atlas
atlas_data, atlas_affine = load_atlas(atlas_path)

# Load the MNI152 NIFTI file
mni152_data, mni152_affine = load_mni152(mni152_path)

# Resample the MNI152 data to match the atlas data
mni152_data_res = nli.resample_to_img(mni152_path, atlas_path).get_fdata()

# Parcellate the resampled MNI152 data using the atlas
parcellated_subcort_data = parcellate_mni152_with_atlas(mni152_data_res, atlas_data)

# Save the parcellated data
parcellated_img = nib.Nifti1Image(parcellated_subcort_data, mni152_affine)
print(f'Shape of the parcellated subcortical image: {parcellated_img.shape}')


'''
# Same with cortical data?

###load and prep cortical data
img = nib.load(os.path.join(data_directory, '4D_rs_fCONF_del_taVNS_sham.nii'))

# Set numpy to print only 2 decimal digits for neatness
np.set_printoptions(precision=2, suppress=True)

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

##transformation
# Transform my image to fsLR
mean_img_fslr_data, _ = transforms.mni152_to_fslr(mean_img_gm, '32k', method='nearest')
mean_img_fslr_data_array = mean_img_fslr_data.darrays[0].data.astype(np.float32)
mean_img_fslr = nlg.GiftiImage(darrays=[nlg.GiftiDataArray(data=mean_img_fslr_data_array)])

# Save the fsLR-transformed image as a Gifti file
nib.save(mean_img_fslr, os.path.join(data_directory, 'mean_img_fslr.gii'))

# Load the Gifti file using nibabel
mean_img_fslr_gifti = nib.load(os.path.join(data_directory, 'mean_img_fslr.gii'))

# Load the atlas
atlas_img = nib.load(atlas_path)

# Resample the Gifti file to match the atlas data using nibabel
mean_img_fslr_resampled = nli.resample_to_img(mean_img_fslr_gifti, atlas_img)

# Get the data from the resampled image
mean_img_fslr_resampled_data = mean_img_fslr_resampled.get_fdata()

##parcellation
# Parcellate the resampled MNI152 data using the atlas
parcellated_cort_data = parcellate_mni152_with_atlas(mean_img_fslr_resampled_data, atlas_data)

# Save the parcellated data
parcellated_img = nib.Nifti1Image(parcellated_cort_data, mni152_affine)
print(f'Shape of the parcellated cortical image: {parcellated_img.shape}')

# TypeError: Data given cannot be loaded because it is not compatible with nibabel format:
'''