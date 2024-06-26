# Correlation of the r-Values (spatial correlations) of the mean image volumetric and the mean image

import os
import numpy as np
import nibabel as nib
import seaborn as sns
import pandas as pd
from nilearn import image as nli
import matplotlib.pyplot as plt
from netneurotools.datasets import fetch_schaefer2018
from neuromaps import transforms, stats
from neuromaps.resampling import resample_images
from neuromaps.nulls import alexander_bloch
from neuromaps.parcellate import Parcellater
from neuromaps.datasets import fetch_annotation
from neuromaps.images import (relabel_gifti, dlabel_to_gifti)


#-- Debugging --#
import os

from scipy.stats import pearsonr

print(os.environ['PATH'])

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
# The output is a tuple of Gifti Images

# Parcellate my image
mean_img_fslr_parc = parc_fsLR.fit_transform(mean_img_fslr, 'fsLR')
# The output is an array


#-------- SPATIAL NULLS OF THE MEAN IMAGE --------#

# Generate nulls
nulls_mean = alexander_bloch(mean_img_fslr_parc, atlas='fsLR', density='32k', parcellation=parcels_fslr_32k)
#print(nulls_mean)
#print(len(nulls_mean))
# Should be 400



#--------- SPATIAL CORRELATIONS FOR MEAN IMAGE --------#

# Fetch annotation
ding2010 = fetch_annotation(source='ding2010')
# Ding 2010 is in MNI space with 1mm density

# Transformation to the fsLR space (sam density as your transformed data: 32k)
ding2010_fslr = transforms.mni152_to_fslr(ding2010, '32k')
# The annotation and your data is now both fsLR 32k

# Parcellate annotation
ding2010_fslr_parc = parc_fsLR.fit_transform(ding2010_fslr, 'fsLR')

# 2) Calculate spatial correlation and p-value
corr_mean, pval_mean = stats.compare_images(mean_img_fslr_parc, ding2010_fslr_parc, nulls=nulls_mean)
#print(f"Correlation for neuromaps annotation and mean image: {corr_mean[0]}")
#print(f"p-value for annotation and mean image: {corr_mean[1]}")
#print(f'Correlation value for annotation and mean image: {corr_mean}')
#print(f'P-value for annotation and mean image: {pval_mean}')


#--------- LOOP FOR SC FOR MEAN IMAGE WITH 11 MAPS ---------#

# List of annotation sources
annotation_sources = ['alarkurtti2015', 'ding2010', 'fazio2016', 'gallezot2010', 'hesse2017',
                      'jaworska2020', 'kaller2017', 'radnakrishnan2018', 'sandiego2015', 'sasaki2012', 'smith2017']

corr_values_mean = []
p_values_mean = []

for source in annotation_sources:
    # Fetch annotation
    annotation = fetch_annotation(source=source)

    # Transformation to the fsLR space (sam density as your transformed data: 32k)
    annotation_fslr = transforms.mni152_to_fslr(annotation, '32k')
    # The annotation and your data is now both fsLR 32k

    # Parcellate annotation
    annotation_fslr_parc = parc_fsLR.fit_transform(annotation_fslr, 'fsLR')

    # 2) Calculate spatial correlation and p-value
    corr_mean, pval_mean = stats.compare_images(mean_img_fslr_parc, annotation_fslr_parc, nulls=nulls_mean)
    #print(f'Correlation value for annotation ({source}) and mean image: {corr_mean}')
    #print(f'P-value for annotation ({source}) and mean image: {pval_mean}')

    corr_values_mean.append(corr_mean)
    p_values_mean.append(pval_mean)
    #print(corr_values_mean)
    #print(pval_mean)


#-------- LOOP FOR SPATIAL CORRELATIONS OF THE MEAN IMAGE WITH THE SUBCORTICAL DATA AND 11 MAPS --------#

annotation_sources = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

corr_values_mean_subcortical = []
p_values_mean_subcortical = []

# Load the mean volumetric image
#mean_orig_img = nib.load(f'/home/leni/Documents/Master/data/combined_mask.nii.gz')


######### test with tian atlas parcellation

# Paths to the atlas and MNI152 NIFTI files
atlas_path = '/home/leni/Tian2020MSA/Tian2020MSA/3T/Subcortex-Only/Tian_Subcortex_S4_3T.nii.gz'
mni152_path = os.path.join(data_directory, 'mean_img_gm.nii')

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
    mni152_img = nib.load(os.path.join(data_directory, 'mean_img_gm.nii'))
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

# Load the atlas
atlas_data, atlas_affine = load_atlas(atlas_path)

# Load the MNI152 NIFTI file
mni152_data, mni152_affine = load_mni152(mni152_path)

print(f"MNI152 data shape: {mni152_data.shape}")
print(f"Atlas data shape: {atlas_data.shape}")

# Resample the MNI152 data to match the atlas data
mni152_data_resampled_img = nli.resample_to_img(nib.Nifti1Image(mni152_data, mni152_affine), nib.Nifti1Image(atlas_data, atlas_affine))
mni152_data_res = mni152_data_resampled_img.get_fdata()

# Parcellate the resampled MNI152 data using the atlas
parcellated_subcort_data = parcellate_mni152_with_atlas(mni152_data_res, atlas_data)

# Save the parcellated data
parcellated_subcort_img = nib.Nifti1Image(parcellated_subcort_data, mni152_affine)
print(f'Shape of the parcellated subcortical image: {parcellated_subcort_img.shape}')

######## test tian end


for source in annotation_sources:
    # Fetch annotation
    annotation = fetch_annotation(source=source)

    # Resample the original data to match the annotation space
    data_res, anno_res = resample_images(src=parcellated_subcort_img, trg=annotation,
                                              src_space='MNI152', trg_space='MNI152',
                                              method='linear', resampling='downsample_only')

    # Calculate spatial correlation and p-value of subcortical data
    corr_mean = stats.compare_images(data_res, anno_res)
    #print(f'Correlation value for annotation ({source}) and mean image: {corr_mean}')

    corr_values_mean_subcortical.append(corr_mean)
    #print(corr_values_mean)

###################################################################
# NEW #

# List of annotation sources
#annotation_sources = ['alarkurtti2015', 'ding2010', 'fazio2016', 'gallezot2010', 'hesse2017',
#                     'jaworska2020', 'kaller2017', 'radnakrishnan2018', 'sandiego2015', 'sasaki2012', 'smith2017']

# r-values mean cortical
corr_values_mean_cortical = corr_values_mean
print(corr_values_mean_cortical)

# r-values mean subcortical
print(corr_values_mean_subcortical)

# eig. unnötig
# Calculate Pearson correlation coefficient
pearson_corr = np.corrcoef(corr_values_mean_cortical, corr_values_mean_subcortical)[0, 1]
#print("Pearson correlation coefficient:", pearson_corr)
# -0.0003843660803493196


'''
#############
# Same for single subject data


#-------- LOOP FOR SC AND P-VALUES FOR ALL SINGLE VOLUMES --------#

# Load all volumes
volumes = []
for i in range(1, 42):
    filename = os.path.join(data_directory, f'volume_{i}.nii')
    # Load the NIfTI file
    img = nib.load(filename)
    # Get the image data as a NumPy array
    data = img.get_fdata()
    # Append the data to the list
    volumes.append(data)

# Convert the list of volumes to a NumPy array
volumes_array = np.array(volumes)

# Load volume
vol_1 = nib.load(os.path.join(data_directory, f'volume_1.nii'))

# Resample and add GM mask
gray_matter_mask_resampled = nli.resample_to_img(gray_matter_mask, vol_1)

# Create a new mask by keeping only non-NaN values in both masks
vol_1_gm_data = np.where(np.isnan(vol_1.get_fdata()), gray_matter_mask_resampled.get_fdata(), vol_1.get_fdata())

# Create a new image with the new mask
vol_1_gm = nib.Nifti1Image(mean_img_gm_data.astype(np.float32), mean_img.affine)

# Save the masked data
nib.save(vol_1_gm, os.path.join(data_directory, 'vol_1_gm.nii'))

# Path to masked data
path_vol_1_gm = os.path.join(data_directory, 'vol_1_gm.nii')

# Transform single volume to fsLR space
vol_1_fslr = transforms.mni152_to_fslr(vol_1_gm, '32k')
# The annotation and your data is now both fsLR 32k

# Parcellate single volume
vol_1_fslr_parc = parc_fsLR.fit_transform(vol_1_fslr, 'fsLR')
#print(f'Shape of the parcellated single volume: {vol_1_fslr_parc.shape}')
#print(f'Shape of the parcellated annotation: {ding2010_fslr_parc.shape}')
# Both should be (400,) for 400 parcellations

# Generate nulls
nulls_single = alexander_bloch(vol_1_fslr_parc, atlas='fsLR', density='32k', parcellation=parcels_fslr_32k)
#print(len(nulls_single))

annotation_sources = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']'''


'''
all_corr_values_single = []
all_p_values_single = []

for source in annotation_sources:
    # Fetch annotation
    anno = fetch_annotation(source=source)

    # Initialize correlation values list
    corr_values_single = []
    p_values_single = []

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
        # Save the masked data
        #nib.save(vol_1_gm, os.path.join(data_directory, 'vol_1_gm.nii'))

        # Transform the single volumes to fsLR
        vol_fslr = transforms.mni152_to_fslr(vol_gm, '32k')
        # Parcellate the single volumes
        vol_fslr_parc = parc_fsLR.fit_transform(vol_fslr, 'fsLR')

        # Transform annotation to fsLR
        anno_fslr = transforms.mni152_to_fslr(anno, '32k')
        # Parcellate annotation
        anno_fslr_parc = parc_fsLR.fit_transform(anno_fslr, 'fsLR')

        # Calculate spatial correlations
        corr, pval = stats.compare_images(vol_fslr_parc, anno_fslr_parc, nulls=nulls_single)
        corr_values_single.append(corr)
        p_values_single.append(pval)

        #print(f"Processing {volume_path}")
        #print(f'r = {corr:.3f}')
        #print("\n")

    # Print the summary array of correlation values
    #print(f"Here are the spatial correlations and the p-values for 41 single volumes with the annotation {source}:")
    #print(np.array(corr_values_single))
    #print(p_values_single)

    # Store the correlation values for this annotation in the list of all correlations
    all_corr_values_single.append(corr_values_single)
    all_p_values_single.append(p_values_single)
    print(len(all_corr_values_single))  # Should print 41
    print(len(all_corr_values_single[0]))  # Should print the number of annotation sources
    print(len(all_p_values_single))  # Should print 41
    print(len(all_p_values_single[0]))  # Should print the number of annotation sources

# Print the list of all correlations
print("Here are the spatial correlations for all annotations:")
print(np.array(all_corr_values_single).shape)

#####
corr_values_single_cortical = all_corr_values_single
#####

# Same for subcortical data:


all_corr_values_single_subcortical = []
all_p_values_single_subcortical = []

for source in annotation_sources:
    # Fetch annotation
    anno = fetch_annotation(source=source)

    # Initialize correlation values list
    corr_values_single_subcortical = []
    p_values_single_subcortical = []

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
        # Save the masked data
        #nib.save(vol_1_gm, os.path.join(data_directory, 'vol_1_gm.nii'))

        # Resample the original data to match the annotation space
        data_res, anno_res = resample_images(src=vol_gm, trg=anno,
                                                  src_space='MNI152', trg_space='MNI152',
                                                  method='linear', resampling='downsample_only')
        print(f'Shape of data_res: {data_res.shape}')
        print(f'Shape of anno_res: {anno_res.shape}')

        # Calculate spatial correlations
        corr = stats.compare_images(data_res, anno_res, metric='pearsonr')
        corr_values_single_subcortical.append(corr)
        #p_values_single_subcortical.append(pval)

        print(f"Processing {volume_path}")
        print(f'r = {corr:.3f}')
        print("\n")

    # Print the summary array of correlation values
    print(f"Here are the spatial correlations and the p-values for 41 single volumes with the annotation {source}:")
    print(np.array(corr_values_single_subcortical))
    #print(p_values_single_subcortical)

    # Store the correlation values for this annotation in the list of all correlations
    all_corr_values_single_subcortical.append(corr_values_single_subcortical)
    #all_p_values_single_subcortical.append(p_values_single_subcortical)

# Print the list of all correlations
print("Here are the spatial correlations for all annotations:")
print(np.array(all_corr_values_single_subcortical).shape)
'''

'''
##########################
# SPAGHETTI PLOT

# Cortical mean r-values (11 maps)
corr_values_mean_cortical = corr_values_mean
print(corr_values_mean_cortical)

# Subcortical mean r-values (11 maps)
print(corr_values_mean_subcortical)

# Cortical single subject r-values (11 maps)
#print(corr_values_single_cortical)

# Subcortical single subject r-values (11 maps)
#print(all_corr_values_single_subcortical)
'''


# Plotting

'''# Version with dots

# Convert lists to numpy arrays
#np_array_subcortical_single = np.array(all_corr_values_single_subcortical)
#np_array_cortical_single = np.array(corr_values_single_cortical)

np_array_subcortical_mean = np.array(corr_values_mean_subcortical)
print(np_array_subcortical_mean)
np_array_cortical_mean = np.array(corr_values_mean_cortical)
print(np_array_cortical_mean)

# Corrected spaghettiplot for mean data

# Add map labels
map_labels = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

# Get the number of maps
num_maps = len(map_labels)

# Calculate the spacing between each line
line_spacing = 1 / 2 * (num_maps + 1)

# Plot mean values and map labels
for i, (mean_value_cortical, mean_value_subcortical, label) in enumerate(zip(np_array_cortical_mean, np_array_subcortical_mean, map_labels)):
    # Plot mean values for cortical data
    plt.plot(mean_value_cortical, (2*i + 1) * line_spacing, 'bo', label=f'Mean Cortical: {mean_value_cortical:.2f}', markersize=3)
    # Plot mean values for subcortical data
    plt.plot(mean_value_subcortical, (2*i + 2) * line_spacing, 'ro', label=f'Mean Subcortical: {mean_value_subcortical:.2f}', markersize=3)
    # Plot map labels on the right side
    plt.text(1.02, (2*i + 1) * line_spacing, label, fontsize=10, verticalalignment='center')
    plt.text(1.02, (2*i + 2) * line_spacing, label, fontsize=10, verticalalignment='center')
    # Plot lines connecting the points
    plt.plot([mean_value_cortical, mean_value_subcortical], [(2 * i + 1) * line_spacing, (2 * i + 2) * line_spacing],
             color='gray')

# Add x-axis label
plt.xlabel('Spatial Correlation Values')

# Add y-axis label
plt.ylabel('Test Categories')

plt.show()

'''


# Convert lists to numpy arrays
np_array_subcortical_mean = np.array(corr_values_mean_subcortical)
np_array_cortical_mean = np.array(corr_values_mean_cortical)

# Add map labels
map_labels = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
              'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

# Get the number of maps
num_maps = len(map_labels)

# Calculate the spacing between each line
line_spacing = 1 / 2 * (num_maps + 1)

'''# Plot mean values and map labels
for i, (mean_value_cortical, mean_value_subcortical, label) in enumerate(zip(np_array_cortical_mean, np_array_subcortical_mean, map_labels)):
    # Plot mean values for cortical data
    plt.plot([(2*i + 1) * line_spacing], [mean_value_cortical], 'bo', markersize=3)
    plt.text((2*i + 1) * line_spacing + 0.01, mean_value_cortical, label, fontsize=10, verticalalignment='center')

    # Plot mean values for subcortical data
    plt.plot([(2*i + 2) * line_spacing], [mean_value_subcortical], 'ro', markersize=3)
    plt.text((2*i + 2) * line_spacing + 0.01, mean_value_subcortical, label, fontsize=10, verticalalignment='center')

    # Plot lines connecting the points
    plt.plot([(2*i + 1) * line_spacing, (2*i + 2) * line_spacing], [mean_value_cortical, mean_value_subcortical], color='gray')
'''



'''
# New corrected spaghettiplot version

# Convert lists to numpy arrays
np_array_subcortical_mean = np.array(corr_values_mean_subcortical)
np_array_cortical_mean = np.array(corr_values_mean_cortical)

# Add map labels
map_labels = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
              'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

# Get the number of maps
num_maps = len(map_labels)

# Calculate the spacing between each line
line_spacing = 1 / (num_maps + 1)

# Plot mean values and map labels
for i, (mean_value_cortical, mean_value_subcortical, label) in enumerate(zip(np_array_cortical_mean, np_array_subcortical_mean, map_labels)):
    # Plot mean values for cortical and subcortical data
    plt.plot([0, 1], [mean_value_cortical, mean_value_subcortical], '-o', color='gray', markersize=3)
    plt.text(0, mean_value_cortical, label, fontsize=10, verticalalignment='bottom', horizontalalignment='right')
    plt.text(1, mean_value_subcortical, label, fontsize=10, verticalalignment='bottom', horizontalalignment='left')

# Add y-axis label
plt.ylabel('Spatial Correlation Values')

# Add x-axis label
plt.xlabel('Test Categories')

# Set x-ticks and labels
plt.xticks([0, 1], ['Cortical', 'Subcortical'])

# Show plot
plt.show()

'''


# Convert lists to numpy arrays

np_array_subcortical_mean = np.array(corr_values_mean_subcortical)
print(np_array_subcortical_mean)
np_array_cortical_mean = np.array(corr_values_mean_cortical)
print(np_array_cortical_mean)

# Correlation calculation
correlation_mean = pearsonr(corr_values_mean_cortical, corr_values_mean_subcortical)[0]
print(f'Here is the correlation value of the mean/group data:{correlation_mean}')

###### CORRECTED SPAGHETTI PLOT #######

x_values = ['Cortical', 'Subcortical']

# Define color categories for brain maps
color_categories = {
    0: [0, 1],
    1: [2],
    2: [3, 4, 5, 6],
    3: [7],
    4: [8],
    5: [9],
    6: [10]
}

# Colors for the color categories
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']


# Plotting
plt.figure(figsize=(12, 6))

for category, color in zip(color_categories.values(), colors):
    for i in category:
        y_values = [corr_values_mean_cortical[i], corr_values_mean_subcortical[i]]
        plt.plot(x_values, y_values, marker='o', color=color, label=f'Map {i+1}')


# Customize plot
plt.title('Spaghetti Plot of r-values')
plt.xlabel('Category')
plt.ylabel('r-values')
plt.grid(True)
plt.legend(title='Brain Maps', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show plot
plt.show()




