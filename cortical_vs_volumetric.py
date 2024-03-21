'''
@author: Lena Wiegert

This code calculates and visualizes the correlation between the mean image of volumetric and cortical tVNS data,
a necessary step to find out if a volumetric approach for data comparison is more suitable. 

'''


import os
import numpy as np
import nibabel as nib
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from nilearn import image as nli
from nilearn import datasets, input_data, surface
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from netneurotools.datasets import fetch_schaefer2018
from neuromaps import transforms
from neuromaps.stats import compare_images
from neuromaps.nulls import alexander_bloch
from neuromaps.parcellate import Parcellater
from neuromaps.datasets import fetch_annotation
from neuromaps.images import (relabel_gifti, dlabel_to_gifti)
from matplotlib.colors import LinearSegmentedColormap


#-- Debugging --#
import os
print(os.environ['PATH'])


#-------- LOAD AND PREP DATA --------#

# Define universal data directory
data_directory = '/home/neuromadlab/tVNS_project/data/'

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

# Import the parcellation maps (Schaefer) in fsLR space

# fsLR32k
parcels_fslr_32k = fetch_schaefer2018('fslr32k')['400Parcels7Networks']
parcels_fslr_32k = dlabel_to_gifti(parcels_fslr_32k)
parcels_fslr_32k = relabel_gifti(parcels_fslr_32k)

# Create parcellaters for fsLR
parc_fsLR = Parcellater(parcels_fslr_32k, 'fslr', resampling_target=None)
#parc_mni152 = Parcellater(parcels_mni152, 'mni152', resampling_target='parcellation')

# Transform my image to fsLR
mean_img_fslr = transforms.mni152_to_fslr(mean_img_gm, '32k', method='nearest')
#print(mean_img_fslr)
# Saving mean_img_fslr, which is a tuple of Gifti images
nib.save(mean_img_fslr[0], os.path.join(data_directory, 'mean_img_fslr_LH.gii'))  # Save left hemisphere
nib.save(mean_img_fslr[1], os.path.join(data_directory, 'mean_img_fslr_RH.gii'))  # Save right hemisphere

# The output is a tuple of Gifti Images

# Parcellate my image
#mean_img_fslr_parc = parc_fsLR.fit_transform(mean_img_fslr, 'fsLR')
# The output is an array


#-------- CORRELATION OF VOLUMETRIC AND CORTICAL MEAN IMAGE --------#

'''# Load Harvard-Oxford atlas (atlas for cortical and subcortical brain data)
atlas_data = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
# The first label correspond to the background
print('The atlas contains {} non-overlapping regions'.format(
    len(atlas_data.labels) - 1))

masker = input_data.NiftiLabelsMasker(atlas_data.maps,
                           labels=atlas_data.labels,
                           standardize=True)

# Parcellate images using Harvard-Oxford atlas
atlas_parcels = input_data.NiftiLabelsMasker(labels_img=atlas_data['maps'], standardize=True)

#harvard_parcellater = Parcellater(atlas_parcels, 'fslr', resampling_target=None, hemi='L')
#mean_img_fslr_parc = harvard_parcellater.fit_transform(mean_img_fslr, 'fsLR')


print(f'Volumetric data: {mean_img_gm}')
print(f'Cortical data: {mean_img_fslr}')


#volumetric_data = atlas_parcels.fit_transform(mean_img_gm)
#cortical_data = atlas_parcels.fit_transform(mean_img_fslr_nii)
#print(cortical_data)


# Extract numerical data from NIfTI images
mean_img_gm_data = mean_img_gm.get_fdata()
mean_img_fslr_data = mean_img_fslr.get_fdata()

# Calculate Pearson correlation coefficient
correlation_coefficient, p_value = pearsonr(mean_img_gm_data, mean_img_fslr_data)
print("Pearson correlation coefficient:", correlation_coefficient)
print("p-value:", p_value)
'''

mskFile = ('/home/neuromadlab/tVNS_project/Tian2020MSA_v1.4/Tian2020MSA/3T/Subcortex-Only/Tian_Subcortex_S2_3T.nii.gz')

atlas_nii = mskFile
atlas_nii = nib.load(mskFile)
atlas_data = atlas_nii.get_fdata()

# Extract numerical data from NIfTI images
mean_img_gm_data = mean_img_gm.get_fdata()

# Split the volumetric mean image into left and right hemispheres
mean_img_gm_LH = mean_img_gm_data[:, :, :mean_img_gm_data.shape[2] // 2]
mean_img_gm_RH = mean_img_gm_data[:, :, mean_img_gm_data.shape[2] // 2:]

# Save the left and right hemisphere images
nib.save(nib.Nifti1Image(mean_img_gm_LH, mean_img_gm.affine), os.path.join(data_directory, 'mean_img_gm_LH.nii'))
nib.save(nib.Nifti1Image(mean_img_gm_RH, mean_img_gm.affine), os.path.join(data_directory, 'mean_img_gm_RH.nii'))

#mean_img_fslr_data = mean_img_fslr.get_fdata()

#mean_img_gm_parc = atlas_data[mean_img_gm_data.astype(int)]
#mean_img_fslr_parc = atlas_data[mean_img_fslr.astype(int)]

# Load gifti files
atlas_file1 = os.path.join(data_directory, 'mean_img_fslr_LH.gii')
atlas_file2 = os.path.join(data_directory, 'mean_img_fslr_RH.gii')

# Load the atlas data from the gifti files
mean_img_fslr_LH = nib.load(atlas_file1).agg_data()
mean_img_fslr_RH = nib.load(atlas_file2).agg_data()

'''parcellated_data = []
for gifti_image in [atlas_file1, atlas_file2]:
    gifti_data = nib.load(gifti_image).darrays[0].data
    parcellated_data.append(atlas_data[gifti_data.astype(int)])'''

tian_atlas = nib.load(mskFile)
tian_atlas_data = tian_atlas.get_fdata()

# Define a masker using the Tian atlas
def tian_masker(img_data):
    # Get unique non-zero values from the Tian atlas data
    labels = np.unique(tian_atlas_data)[1:]
    # Iterate over each label and create a mask for it
    masks = [img_data == label for label in labels]
    # Stack the masks along the last dimension
    mask = np.stack(masks, axis=-1)
    return mask

# Parcellate volumetric data
cortical_LH_parc_data = tian_masker(mean_img_gm_LH)
cortical_RH_parc_data = tian_masker(mean_img_gm_RH)
print(cortical_RH_parc_data.shape)

subcortical_LH_parc_data = tian_masker(mean_img_fslr_LH)
subcortical_RH_parc_data = tian_masker(mean_img_fslr_RH)
print(subcortical_RH_parc_data.shape)

# Convert boolean arrays to integer or float arrays
cortical_LH_parc_data = cortical_LH_parc_data.astype(np.float32)  # Convert to float32
cortical_RH_parc_data = cortical_RH_parc_data.astype(np.float32)  # Convert to float32
subcortical_LH_parc_data = subcortical_LH_parc_data.astype(np.float32)  # Convert to float32
subcortical_RH_parc_data = subcortical_RH_parc_data.astype(np.float32)  # Convert to float32

# Create NIfTI image objects
cortical_LH_img = nib.Nifti1Image(cortical_LH_parc_data, affine=None)
cortical_RH_img = nib.Nifti1Image(cortical_RH_parc_data, affine=None)
subcortical_LH_img = nib.Nifti1Image(subcortical_LH_parc_data, affine=None)
subcortical_RH_img = nib.Nifti1Image(subcortical_RH_parc_data, affine=None)

# Calculate Pearson correlation coefficient between cortical data and left hemisphere
#correlation_coefficient_LH, p_value_LH = pearsonr(subcortical_LH_parc_data_res, cortical_LH_parc_data)
corr = compare_images(subcortical_LH_img, cortical_LH_img, metric='pearsonr')
# numpy.exceptions.AxisError: axis 3 is out of bounds for array of dimension 3

# Calculate Pearson correlation coefficient between cortical data and right hemisphere
#correlation_coefficient_RH, p_value_RH = pearsonr(subcortical_RH_parc_data, cortical_RH_parc_data)
# ValueError: x and y must have the same length.

'''
print("Left Hemisphere Pearson correlation coefficient:", correlation_coefficient_LH)
print("Left Hemisphere p-value:", p_value_LH)

print("Right Hemisphere Pearson correlation coefficient:", correlation_coefficient_RH)
print("Right Hemisphere p-value:", p_value_RH)
'''

# Probleme mit den Shapes

# Problem: Cortical brain image is split in left and right hemisphere (tuple of gifti), volumetric brain image is not
# Parcellation does not work with tuple
print(f'The mean image has been parcellated with the Melbourne Subcortical Atlas.')
print(mean_img_gm_parc)
# Bigger problem: Process finished with exit code 137 (interrupted by signal 9:SIGKILL)
# Memory error?

