# Author: Lena Wiegert
# This code is part of the Master Thesis "Placing the Effects of tVNS into Neurobiological Context"

'''
Ths code creates spatial null models of the cortical maps of tVNS-induced changes in relation to receptor maps on group
and individual level (Figure 2B and C). The whole-brain data and the receptor maps (annotations) are transformed to surface-based space
(fsLR) and parcellated with the Human Cerebral Cortex Atlas (Schaefer2018). The spatial correlation between tVNS-induced
changes and receptor maps were calculated and plotted with the spatial nulls (based on the Alexander-Bloch Null Model)
in heatmaps on individual level.
'''


import os
import numpy as np
import nibabel as nib
import seaborn as sns
import pandas as pd
from nilearn import image as nli
import matplotlib.pyplot as plt
from netneurotools.datasets import fetch_schaefer2018
from neuromaps import transforms, stats
from neuromaps.nulls import alexander_bloch
from neuromaps.parcellate import Parcellater
from neuromaps.datasets import fetch_annotation
from neuromaps.images import (relabel_gifti, dlabel_to_gifti)


#-- Debugging --#
import os
print(os.environ['PATH'])


#-------- LOAD AND PREP DATA --------#

# Define universal data directory
#data_directory = '/home/leni/Documents/Master/data/'
data_directory = '/home/neuromadlab/tVNS_project/data/'

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

# Transform my image to fsLR
group_img_fslr = transforms.mni152_to_fslr(group_img_gm, '32k', method='nearest')
print(group_img_fslr)
# The output is a tuple of Gifti Images

# Parcellate my image
group_img_fslr_parc = parc_fsLR.fit_transform(group_img_fslr, 'fsLR')
# The output is an array


#-------- SPATIAL NULL MODEL ON GROUP LEVEL --------#

# Generate nulls
nulls_group = alexander_bloch(group_img_fslr_parc, atlas='fsLR', density='32k', parcellation=parcels_fslr_32k)
print(nulls_group)
print(len(nulls_group))
# Should be 400


#--------- SPATIAL CORRELATIONS ON GROUP LEVEL --------#

# Fetch annotation
ding2010 = fetch_annotation(source='ding2010')
# Ding 2010 is in MNI space with 1mm density

# Transformation to the fsLR space (sam density as your transformed data: 32k)
ding2010_fslr = transforms.mni152_to_fslr(ding2010, '32k')
# The annotation and your data is now both fsLR 32k

# Parcellate annotation
ding2010_fslr_parc = parc_fsLR.fit_transform(ding2010_fslr, 'fsLR')

# 2) Calculate spatial correlation and p-value
corr_group, pval_group = stats.compare_images(group_img_fslr_parc, ding2010_fslr_parc, nulls=nulls_group)
#print(f"Correlation for neuromaps annotation and group image: {corr_group[0]}")
#print(f"p-value for annotation and group image: {corr_group[1]}")
print(f'Correlation value for annotation and group image: {corr_group}')
print(f'P-value for annotation and group image: {pval_group}')


#--------- SPATIAL CORRELATIONS ON GROUP LEVEL ---------#

# List of annotation sources
annotation_sources = ['alarkurtti2015', 'ding2010', 'fazio2016', 'gallezot2010', 'hesse2017',
                      'jaworska2020', 'kaller2017', 'radnakrishnan2018', 'sandiego2015', 'sasaki2012', 'smith2017']

corr_values_group = []
p_values_group = []

for source in annotation_sources:
    # Fetch annotation
    annotation = fetch_annotation(source=source)

    # Transformation to the fsLR space (sam density as your transformed data: 32k)
    annotation_fslr = transforms.mni152_to_fslr(annotation, '32k')
    # The annotation and your data is now both fsLR 32k

    # Parcellate annotation
    annotation_fslr_parc = parc_fsLR.fit_transform(annotation_fslr, 'fsLR')

    # 2) Calculate spatial correlation and p-value
    corr_group, pval_group = stats.compare_images(group_img_fslr_parc, annotation_fslr_parc, nulls=nulls_group)
    print(f'Correlation value for annotation ({source}) and group image: {corr_group}')
    print(f'P-value for annotation ({source}) and group image: {pval_group}')

    corr_values_group.append(corr_group)
    p_values_group.append(pval_group)
    print(corr_values_group)
    print(pval_group)


#------------ SPATIAL CORRELATIONS ON INDIVIDUAL LEVEL ------------#

# Load volume
vol_1 = nib.load(os.path.join(data_directory, f'volume_1.nii'))
# Resample and add GM mask
gray_matter_mask_resampled = nli.resample_to_img(gray_matter_mask, vol_1)
# Create a new mask by keeping only non-NaN values in both masks
vol_1_gm_data = np.where(np.isnan(vol_1.get_fdata()), gray_matter_mask_resampled.get_fdata(), vol_1.get_fdata())
# Create a new image with the new mask
vol_1_gm = nib.Nifti1Image(group_img_gm_data.astype(np.float32), group_img.affine)
# Save the masked data
nib.save(vol_1_gm, os.path.join(data_directory, 'vol_1_gm.nii'))
# Path to masked data
path_vol_1_gm = os.path.join(data_directory, 'vol_1_gm.nii')

# Transform individual volume to fsLR space
vol_1_fslr = transforms.mni152_to_fslr(vol_1_gm, '32k')
# The annotation and your data is now both fsLR 32k

# Parcellate individual volume
vol_1_fslr_parc = parc_fsLR.fit_transform(vol_1_fslr, 'fsLR')
# Double check parcellation
print(f'Shape of the parcellated individual volume: {vol_1_fslr_parc.shape}')
print(f'Shape of the parcellated annotation: {ding2010_fslr_parc.shape}')
# Both should be (400,) for 400 parcellations

# Generate nulls
nulls_individual = alexander_bloch(vol_1_fslr_parc, atlas='fsLR', density='32k', parcellation=parcels_fslr_32k)

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

# List of annotation sources
annotation_sources = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

all_corr_values_individual = []
all_p_values_individual = []

for source in annotation_sources:
    # Fetch annotation
    anno = fetch_annotation(source=source)

    # Initialize correlation values list
    corr_values_individual = []
    p_values_individual = []
    volume_numbers = []  # List to store volume numbers

    # Process each volume
    for i in range(1, 42):
        volume_path = os.path.join(data_directory, f'volume_{i}.nii')
        volume = nib.load(volume_path)
        volume_numbers.append(i)  # Store the volume number

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

        # Transform the individual volumes to fsLR
        vol_fslr = transforms.mni152_to_fslr(vol_gm, '32k')
        # Parcellate the individual volumes
        vol_fslr_parc = parc_fsLR.fit_transform(vol_fslr, 'fsLR')

        # Transform annotation to fsLR
        anno_fslr = transforms.mni152_to_fslr(anno, '32k')
        # Parcellate annotation
        anno_fslr_parc = parc_fsLR.fit_transform(anno_fslr, 'fsLR')

        # Calculate spatial correlations
        corr, pval = stats.compare_images(vol_fslr_parc, anno_fslr_parc, nulls=nulls_individual)
        corr_values_individual.append(corr)
        p_values_individual.append(pval)

        print(f"Processing {volume_path}")
        print(f'r = {corr:.3f}')
        print("\n")

    # Print the summary array of correlation values
    print(f"Here are the spatial correlations and the p-values for 41 individual volumes with the annotation {source}:")
    print(np.array(corr_values_individual))
    print(p_values_individual)

    # Store the correlation values for this annotation in the list of all correlations
    all_corr_values_individual.append(corr_values_individual)
    all_p_values_individual.append(p_values_individual)
    print(len(all_corr_values_individual))  # Should print 41
    print(len(all_p_values_individual))  # Should print 41

# Print the list of all correlations
print("Here are the spatial correlations for all annotations:")
print(np.array(all_corr_values_individual).shape)


#--------- PLOTTING ON INDIVIDUAL LEVEL - RECEPTOR-SPECIFIC HEATMAPS ---------#

'''
This part of the code plots one figure with seperate heatmaps of one receptor type.
In this example, the maps for dopamine are plotted. If you want the plots for the noradrenaline (NE) and serotonin 
receptors, please change the start_index and the end_index accordingly.
For NE: start_index = 0, end_index = 2
For serotonin: start_index = 9, end_index = 12
'''

# Plotting only the dopamine receptor maps:
# Calculate the number of rows and columns for the subplot grid
num_maps = len(annotation_sources)
start_index = 2  # Index of the starting plot (0-indexed)
end_index = 8  # Index of the ending plot (0-indexed, exclusive)

# Determine the number of plots to be included
num_plots = end_index - start_index

# Calculate the number of columns in the grid
num_cols = 1  # Number of columns in the grid to ensure all plots are in a single column

# Calculate the number of rows needed
num_rows = (num_plots + num_cols - 1) // num_cols

# Create a new figure with the appropriate size
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8 * num_rows))

corr_values_all = []

# Loop through each brain map and corresponding axis in the grid
for idx in range(start_index, end_index):
    row_idx = (idx - start_index) // num_cols
    ax = axes[row_idx] if num_rows > 1 else axes  # Adjust for single column layout

    source = annotation_sources[idx]

    # Fetch annotation
    anno = fetch_annotation(source=source)
    # Initialize correlation values list
    corr_values_individual = []
    p_values_individual = []
    volume_numbers = []  # List to store volume numbers

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
        # Parcellate the individual volumes
        vol_fslr_parc = parc_fsLR.fit_transform(vol_fslr, 'fsLR')

        # Transform annotation to fsLR
        anno_fslr = transforms.mni152_to_fslr(anno, '32k')
        # Parcellate annotation
        anno_fslr_parc = parc_fsLR.fit_transform(anno_fslr, 'fsLR')

        # Calculate spatial correlations
        corr, pval = stats.compare_images(vol_fslr_parc, anno_fslr_parc, nulls=nulls_individual)
        corr_values_individual.append(corr)
        p_values_individual.append(pval)

        volume_numbers.append((i, corr))  # Store the volume number and its corresponding correlation value

        print(f"Processing {volume_path}")
        print(f'r = {corr:.3f}')
        print("\n")

    corr_values_all.extend(corr_values_individual)
    # Calculate the highest and lowest total correlation values
    min_corr = min(corr_values_all)
    max_corr = max(corr_values_all)

    # Create a DataFrame for correlation values for the current brain map
    df_corr_individual = pd.DataFrame({'Correlation Values': corr_values_individual})

    df_p_values_individual = pd.DataFrame({'p-Values': p_values_individual})

    # Sort correlation values from lowest to highest
    df_corr_individual_sorted = df_corr_individual.sort_values(by='Correlation Values')

    # Sort p-values based on the sorted correlation values
    df_p_values_individual_sorted = df_p_values_individual.loc[df_corr_individual_sorted.index]

    # Plot the heatmap of sorted correlation values for the current brain map
    sns.heatmap(df_corr_individual_sorted.transpose(), annot=False, fmt='.2f', cmap='coolwarm', cbar=True, center=(min_corr + max_corr) / 2,
                ax=ax, vmin=min_corr, vmax=max_corr)
    ax.set_title(f'{source}')
    ax.set_xlabel('Volume Number')
    ax.set_ylabel('Correlation Value')

    # Loop through each cell and add the p-value as annotation text if it's significant
    for i in range(df_p_values_individual_sorted.shape[0]):
        if (df_p_values_individual_sorted.iloc[i] <= 0.05).any():
            ax.text(i + 0.5, 0.5, '*', ha='center', va='center', color='black', fontsize=10)

    # Rotate x-axis ticks
    ax.tick_params(axis='x', rotation=45, labelsize=6)

    # Sort volume numbers based on the correlation values
    volume_numbers_sorted = sorted(volume_numbers, key=lambda x: x[1])

    # Extract only the sorted volume numbers
    sorted_volume_numbers = [volume[0] for volume in volume_numbers_sorted]

    # Set x-ticks and corresponding labels for each volume
    ax.set_xticks(np.arange(len(df_corr_individual)))
    ax.set_xticklabels(sorted_volume_numbers[:len(df_corr_individual)])  # Set sorted volume numbers as xticklabels

    # Extend the list with correlation values for the current brain map
    corr_values_all.extend(corr_values_individual)

# Remove empty subplots if any
for i in range(num_plots, num_cols * num_rows):
    fig.delaxes(axes.flatten()[i])

# Adjust layout to prevent overlapping and increase space between rows
plt.tight_layout(pad=12.0)  # Adjust padding for better layout

plt.suptitle('Correlations of cortical maps of tVNS-induced changes and dopamine receptor maps (individual level)', fontsize=20, fontweight='bold', y=1.1)
# Save the plot as an image
plt.savefig('heatmap_combined.png')

# Show the plot
plt.show()


#------------- SAVE FIGURE ------------#

# Define the filename for the figure
figure_filename = 'cortical_ind_heatmaps_final.png'
# Construct the full file path
figure_path = os.path.join(data_directory, figure_filename)
# Save the figure
fig.savefig(figure_path, dpi=300, bbox_inches='tight')

