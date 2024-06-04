'''
@author: Lena Wiegert

This code prepares the data for further analysis by creating seperate files for each individual dataset (i.e. for
each participant, named by their ID). Further, a mean image is created and different plotting options are tested.
The spatial correlations of the maps of tVNS-induced changes with 11 receptor maps are calculated on group and
individual level.
'''

import os
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from nilearn import image as nli
from netneurotools.datasets import fetch_schaefer2018
from neuromaps import transforms, stats
from neuromaps.nulls import alexander_bloch
from neuromaps.parcellate import Parcellater
from neuromaps.datasets import fetch_annotation
from neuromaps.images import (relabel_gifti, dlabel_to_gifti)
from scipy.stats import pearsonr

#-------- LOAD AND PREP DATA --------#

# Define universal data directory
data_directory = '/home/leni/Documents/Master/data/'
#data_directory = '/home/neuromadlab/tVNS_project/data/'

img = nib.load(os.path.join(data_directory, '4D_rs_fCONF_del_taVNS_sham.nii'))

# Set numpy to print only 2 decimal digits for neatness
np.set_printoptions(precision=2, suppress=True)
# Array proxy to create the image object without immediately loading all the array data from disk
nib.is_proxy(img.dataobj)

# Create group image
group_img = nli.mean_img(img)

# Replace non-finite values with a gray matter mask
gray_matter_mask_file = os.path.join(data_directory, 'out_GM_p_0_15.nii')
gray_matter_mask = nib.load(gray_matter_mask_file)

# Resample gray matter mask to match the resolution of group_img
gray_matter_mask_resampled = nli.resample_to_img(gray_matter_mask, group_img)

# Ensure both masks have the same shape
if not np.all(gray_matter_mask_resampled.shape == group_img.shape):
    raise ValueError('Shape of input volume is incompatible.')

# Create a new mask by keeping only non-NaN values in both masks
group_img_gm_data = np.where(np.isnan(group_img.get_fdata()), gray_matter_mask_resampled.get_fdata(), group_img.get_fdata())

# Create a new image with the new mask
group_img_gm = nib.Nifti1Image(group_img_gm_data.astype(np.float32), group_img.affine)

# Make sure the affines are the same
group_img_gm = nli.resample_to_img(group_img_gm, img)

# Extract data from Nifti1Image
group_img_data = group_img_gm.get_fdata()

# Save the masked data
nib.save(group_img_gm, os.path.join(data_directory, 'group_img_gm.nii'))

# Path to masked data
path_group_img_gm = os.path.join(data_directory, 'group_img_gm.nii')



######## CORTICAL DATA - GROUP LEVEL ########

#-------- CORTICAL PARCELLATION --------#

# Import the parcellation maps (Schaefer) in fsLR space
# fsLR32k
parcels_fslr_32k = fetch_schaefer2018('fslr32k')['400Parcels7Networks']
parcels_fslr_32k = dlabel_to_gifti(parcels_fslr_32k)
parcels_fslr_32k = relabel_gifti(parcels_fslr_32k)
# Create parcellaters for fsLR
parc_fsLR = Parcellater(parcels_fslr_32k, 'fslr', resampling_target=None)
# Transform my image to fsLR
group_img_fslr = transforms.mni152_to_fslr(group_img_gm, '32k', method='nearest')
print(group_img_fslr)
# The output is a tuple of Gifti Images

# Parcellate group image
group_img_fslr_parc = parc_fsLR.fit_transform(group_img_fslr, 'fsLR')
# The output is an array

#-------- SPATIAL NULLS --------#

# Generate nulls
nulls_group = alexander_bloch(group_img_fslr_parc, atlas='fsLR', density='32k', parcellation=parcels_fslr_32k)
print(nulls_group)
print(len(nulls_group))
# Should be 400

# List of annotation sources
annotation_sources = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

corr_values_group = []
p_values_group = []

for source in annotation_sources:
    # Fetch annotation
    annotation = fetch_annotation(source=source)

    # Transformation to the fsLR space (same density as the transformed data: 32k)
    annotation_fslr = transforms.mni152_to_fslr(annotation, '32k')
    # The annotation and the data are now both fsLR 32k

    # Parcellate annotation
    annotation_fslr_parc = parc_fsLR.fit_transform(annotation_fslr, 'fsLR')

    # 2) Calculate spatial correlation and p-value
    corr_group, pval_group = stats.compare_images(group_img_fslr_parc, annotation_fslr_parc, nulls=nulls_group)
    #print(f'Correlation value for annotation ({source}) and group image: {corr_group}')
    #print(f'P-value for annotation ({source}) and group image: {pval_group}')

    corr_values_group.append(corr_group)
    p_values_group.append(pval_group)
    #print(corr_values_group)
    #print(pval_group)

# Convert list to numpy array
corr_values_group_array = np.array(corr_values_group)
file_path = os.path.join(data_directory, 'corr_values_group_cortical.npy')
np.save(file_path, corr_values_group_array)
print(corr_values_group_array.shape)


######## CORTICAL DATA - INDIVIDUAL LEVEL ########

# Initialize correlation values list
corr_values_cortical_individual = []
# Initialize correlation values dictionary for each brain map
corr_values_cortical_individual_maps = {}

# List of brain maps
brain_maps = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
              'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

# Loop through each brain map
for brain_map in brain_maps:
    print(f"Processing brain map: {brain_map}")

    # Initialize correlation values list for the current brain map
    corr_values_cortical_individual = []

    # Load annotation data for the current brain map
    anno = fetch_annotation(source=brain_map)
    anno_img = nib.load(anno)
    anno_data = anno_img.get_fdata()
    anno_affine = anno_img.affine

    # Process each volume
    for i in range(1, 42):
        volume_path = os.path.join(data_directory, f'volume_{i}.nii')
        volume = nib.load(volume_path)
        # Resample and add GM mask
        gray_matter_mask_resampled = nli.resample_to_img(gray_matter_mask, volume)
        # Create a new mask by keeping only non-NaN values in both masks
        vol_gm_data = np.where(np.isnan(volume.get_fdata()), gray_matter_mask_resampled.get_fdata(), volume.get_fdata())
        # Create a new image with the new mask
        vol_gm = nib.Nifti1Image(vol_gm_data.astype(np.float32), volume.affine)

        # Transform the individual volumes to fsLR
        vol_fslr = transforms.mni152_to_fslr(vol_gm, '32k')
        # Parcellate the individual volumes with Schaefer
        vol_fslr_parc = parc_fsLR.fit_transform(vol_fslr, 'fsLR')

        # Transform annotation to fsLR
        anno_fslr = transforms.mni152_to_fslr(anno, '32k')
        # Parcellate annotation with Schaefer
        anno_fslr_parc = parc_fsLR.fit_transform(anno_fslr, 'fsLR')

        # Calculate spatial correlations
        corr = stats.compare_images(vol_fslr_parc, anno_fslr_parc)
        corr_values_cortical_individual.append(corr)

    # Save the correlation values for the current brain map
    corr_values_cortical_individual_maps[brain_map] = corr_values_cortical_individual

    print(f'Here are the correlation values for cortical individual subject data with {brain_map}: {corr_values_cortical_individual}')

# Save the correlation values for the subcortical individual maps
print(corr_values_cortical_individual_maps)
file_path_cort = os.path.join(data_directory, 'corr_values_cortical_individual_maps.npy')
np.save(file_path_cort, corr_values_cortical_individual_maps)


######## SUBCORTICAL DATA - GROUP LEVEL ########

#-------- SPATIAL CORRELATIONS --------#

annotation_sources = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

corr_values_group_subcortical = []
p_values_group_subcortical = []

# Load the group volumetric image
group_orig_img = nib.load(f'{data_directory}combined_mask.nii.gz')

#-------- SUBCORTICAL PARCELLATION --------#

# Load Nifti brain atlas file
#atlas_path = '/home/neuromadlab/Tian2020MSA_v1.4/Tian2020MSA/3T/Cortex-Subcortex/MNIvolumetric/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S4_3T_MNI152NLin2009cAsym_2mm.nii.gz'
atlas_path = '/home/leni/Tian2020MSA_v1.4/Tian2020MSA/3T/Cortex-Subcortex/MNIvolumetric/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S4_3T_MNI152NLin2009cAsym_2mm.nii.gz'
atlas_img = nib.load(atlas_path)

# Initialize Parcellater with the Nifti brain atlas
parcellater = Parcellater(parcellation=atlas_img,  space='MNI152')

# Fit the Parcellater
parcellater.fit()

# Parcellate the resampled MNI152 data using the atlas
parcellated_subcort_group_data = parcellater.transform(group_orig_img, space='MNI152')
print(parcellated_subcort_group_data)

# Check the shape of the parcellated data
print(f'Shape of the parcellated subcortical data: {parcellated_subcort_group_data.shape}')

# r-values group cortical
corr_values_group_cortical = corr_values_group
print(corr_values_group_cortical)

#-------- SPATIAL NULLS --------#

for source in annotation_sources:
    # Fetch annotation
    annotation = fetch_annotation(source=source)
    # Parcellate annotation with Tian
    annotation_mni152_parc = parcellater.transform(annotation, space='MNI152')

    # Calculate spatial correlation and p-value of subcortical data
    corr_group = stats.compare_images(parcellated_subcort_group_data, annotation_mni152_parc)

    # Append to list
    corr_values_group_subcortical.append(corr_group)
    #print(corr_values_group)

# r-values group subcortical
print(corr_values_group_subcortical)
corr_values_group_subcortical_array = np.array(corr_values_group_subcortical)
print(f'Shape of parcellated subcortical group data array: {corr_values_group_subcortical_array.shape}')
file_path = os.path.join(data_directory, 'corr_values_group_subcortical.npy')
np.save(file_path, corr_values_group_subcortical_array)

######## SUBCORTICAL DATA - INDIVIDUAL LEVEL ########

# Initialize correlation values dictionary for each brain map
corr_values_subcortical_individual_maps = {}

# Loop through each brain map
for brain_map in brain_maps:
    print(f"Processing brain map: {brain_map}")

    # Initialize correlation values list for the current brain map
    corr_values_subcortical_individual = []

    # Load annotation data for the current brain map
    anno = fetch_annotation(source=brain_map)
    anno_img = nib.load(anno)
    anno_data = anno_img.get_fdata()

    # Process each participant
    for i in range(1, 42):
        volume_path = os.path.join(data_directory, f'volume_{i}.nii')
        volume = nib.load(volume_path)
        #print(f"Processing volume {i}: {volume_path}")  # Debugging print statement
        # Resample and add GM mask
        gray_matter_mask_resampled = nli.resample_to_img(gray_matter_mask, volume)
        # Create a new mask by keeping only non-NaN values in both masks
        vol_gm_data = np.where(np.isnan(volume.get_fdata()), gray_matter_mask_resampled.get_fdata(), volume.get_fdata())
        # Create a new image with the new mask
        vol_gm = nib.Nifti1Image(vol_gm_data.astype(np.float32), volume.affine)
        vol_gm_affine = vol_gm.affine

        # Parcellate the single volumes with Tian
        parcellated_subcort_single_data = parcellater.transform(vol_gm, space='MNI152')
        print(f'Shape of the parcellated subcortical data: {parcellated_subcort_single_data.shape}')

        # Parcellate annotation data with Tian
        anno_mni152_parc = parcellater.transform(anno_img, space='MNI152')

        # Calculate spatial correlations
        corr = stats.compare_images(parcellated_subcort_single_data, anno_mni152_parc)
        corr_values_subcortical_single.append(corr)
        #print(f"Correlation value for volume {i}: {corr}")

    # Save the correlation values for the current brain map
    corr_values_subcortical_single_maps[brain_map] = corr_values_subcortical_single

    print(f'Here are the correlation values for subcortical single subject data with {brain_map}: {corr_values_subcortical_single}')
    #print(len(corr_values_subcortical_single))

# Save the correlation values for the subcortical single maps
print(corr_values_subcortical_single_maps)
file_path_subcort = os.path.join(data_directory, 'corr_values_subcortical_single_maps.npy')
np.save(file_path_subcort, corr_values_subcortical_single_maps)

