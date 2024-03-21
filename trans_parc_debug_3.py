'''
@author: Lena Wiegert

This code transforms MNI152 data to fsLR space and parcellates it with the Schaefer atlas.
It is based on this workshop: https://www.youtube.com/watch?v=pc8zMMTLxmA

The code further visualizes the significance of the spatial correlations for mean and individual data.
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
print(mean_img_fslr)
# The output is a tuple of Gifti Images

# Parcellate my image
mean_img_fslr_parc = parc_fsLR.fit_transform(mean_img_fslr, 'fsLR')
# The output is an array


#-------- SPATIAL NULLS OF THE MEAN IMAGE --------#

# Generate nulls
nulls_mean = alexander_bloch(mean_img_fslr_parc, atlas='fsLR', density='32k', parcellation=parcels_fslr_32k)
print(nulls_mean)
print(len(nulls_mean))
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
print(f'Correlation value for annotation and mean image: {corr_mean}')
print(f'P-value for annotation and mean image: {pval_mean}')


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
    print(f'Correlation value for annotation ({source}) and mean image: {corr_mean}')
    print(f'P-value for annotation ({source}) and mean image: {pval_mean}')

    corr_values_mean.append(corr_mean)
    p_values_mean.append(pval_mean)
    print(corr_values_mean)
    print(pval_mean)

#-------- SC + P-VALUE PLOTTING MEAN IMAGE --------#

# Assuming `corr_values_mean` and `p_values_mean` are lists of correlation and p-values respectively
data = {'Annotation Source': annotation_sources,
        'Correlation Value': corr_values_mean,
        'P-Value': p_values_mean}

df = pd.DataFrame(data)

# Sort the DataFrame by P-Value in ascending order
df_sorted = df.sort_values(by='P-Value')

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df_sorted.set_index('Annotation Source'), annot=True, fmt='.2g', cmap='coolwarm', ax=ax, center=0.05)
plt.xticks(rotation=45)
plt.title('Correlation Values and P-Values for Each Annotation Source (Sorted by P-Value)')
plt.show()




#------------ SPATIAL CORRELATIONS FOR ALL 41 FILES ------------#

# Each volume of my data compared to Hesse 2017

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
print(f'Shape of the parcellated single volume: {vol_1_fslr_parc.shape}')
print(f'Shape of the parcellated annotation: {ding2010_fslr_parc.shape}')
# Both should be (400,) for 400 parcellations

# Generate nulls
nulls_single = alexander_bloch(vol_1_fslr_parc, atlas='fsLR', density='32k', parcellation=parcels_fslr_32k)
print(len(nulls_single))

# Calculate SC
corr_single, pval_single = stats.compare_images(vol_1_fslr_parc, ding2010_fslr_parc, nulls=nulls_single)
print(f'Correlation value for annotation and single volume: {corr_single}')
print(f'P-value for annotation and single volume: {pval_single}')


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


# List of annotation sources
#annotation_sources = ['alarkurtti2015', 'ding2010', 'fazio2016', 'gallezot2010', 'hesse2017',
#                      'jaworska2020', 'kaller2017', 'radnakrishnan2018', 'sandiego2015', 'sasaki2012', 'smith2017']
annotation_sources = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

all_corr_values_single = []
all_p_values_single = []

for source in annotation_sources:
    # Fetch annotation
    anno = fetch_annotation(source=source)

    # Initialize correlation values list
    corr_values_single = []
    p_values_single = []
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

        print(f"Processing {volume_path}")
        print(f'r = {corr:.3f}')
        print("\n")

    # Print the summary array of correlation values
    print(f"Here are the spatial correlations and the p-values for 41 single volumes with the annotation {source}:")
    print(np.array(corr_values_single))
    print(p_values_single)

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




'''
# One big heatmap
#--------- SC + P-VALUE PLOTTING OF SINGLE SUBJECT DATA ---------#

# Create a new array with the p-values only
p_values_only = np.array(all_p_values_single)

# Replace the p-values that are not significant (greater than 0.05) with NaN
p_values_only[p_values_only > 0.05] = np.nan


# Create a DataFrame for correlation values
df_corr = pd.DataFrame(all_corr_values_single, columns=np.arange(1, 42), index=annotation_sources)

# Create a DataFrame for correlation values
df_p_values = pd.DataFrame(all_p_values_single, columns=np.arange(1, 42), index=annotation_sources)

# Transpose the DataFrame for correlation values
df_corr_transposed = df_corr.transpose()

# Transpose the DataFrame for p-values
df_p_values_transposed = df_p_values.transpose()

# Initialize the figure
plt.figure(figsize=(12, 6))

# Plot the heatmap of correlation values
plt.subplot(1, 2, 1)
sns.heatmap(df_corr_transposed, annot=False, fmt='.2f', cmap='coolwarm', cbar=False, center=0.05)

# Loop through each cell and add the p-value as annotation text if it's significant
for i in range(df_p_values_transposed.shape[0]):
    for j in range(df_p_values_transposed.shape[1]):
        if df_p_values_transposed.iloc[i, j] <= 0.05:
            plt.text(j + 0.5, i + 0.5, f'{df_p_values_transposed.iloc[i, j]:.3f}',
                     ha='center', va='center', color='black', fontsize=7)


# Initialize the figure
plt.figure(figsize=(12, 6))

# Plot the heatmap of correlation values
plt.subplot(1, 2, 1)
sns.heatmap(df_corr, annot=False, fmt='.2f', cmap='coolwarm', cbar=False, center=0.05)

# Loop through each cell and add the p-value as annotation text if it's significant
for i in range(df_p_values.shape[0]):
    for j in range(df_p_values.shape[1]):
        if df_p_values.iloc[i, j] <= 0.05:''''''
            plt.text(j + 0.5, i + 0.5, f'{df_p_values.iloc[i, j]:.3f}',
                     ha='center', va='center', color='black', fontsize=7)

plt.title('Correlation Values with Significant P-Values')
plt.show()
'''


'''
# Single heatmaps
# Calculate the number of rows and columns for the subplot grid
num_maps = len(annotation_sources)
num_cols = 3  # Number of columns in the grid
num_rows = (num_maps + num_cols - 1) // num_cols  # Calculate the number of rows needed

# Create a new figure with the appropriate size
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8 * num_rows))

# Loop through each brain map and corresponding axis in the grid
for idx, source in enumerate(annotation_sources):
    row_idx = idx // num_cols
    col_idx = idx % num_cols
    ax = axes[row_idx, col_idx]  # Get the axis corresponding to the current brain map

    # Fetch annotation
    anno = fetch_annotation(source=source)

    # Initialize correlation values list
    corr_values_single = []
    p_values_single = []
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

        volume_numbers.append(i)  # Store the volume number

        print(f"Processing {volume_path}")
        print(f'r = {corr:.3f}')
        print("\n")

    # Create a DataFrame for correlation values for the current brain map
    df_corr_single = pd.DataFrame({'Correlation Values': corr_values_single})

    df_p_values_single = pd.DataFrame({'p-Values':p_values_single})

    # Sort correlation values from lowest to highest
    df_corr_single_sorted = df_corr_single.sort_values(by='Correlation Values')

    # Sort p-values based on the sorted correlation values
    df_p_values_single_sorted = df_p_values_single.loc[df_corr_single_sorted.index]

    # Plot the heatmap of sorted correlation values for the current brain map
    sns.heatmap(df_corr_single_sorted.transpose(), annot=False, fmt='.2f', cmap='coolwarm', cbar=True, center=0.05,
                ax=ax)
    ax.set_title(f'Correlation Values with Significant P-Values for {source}')
    ax.set_xlabel('Volume Number')
    if col_idx == 0:  # Add y-axis label only for the first column
        ax.set_ylabel('Correlation Value')
    else:
        ax.set_ylabel('')  # Remove y-axis label for other columns

    # Loop through each cell and add the p-value as annotation text if it's significant
    #for i in range(df_corr_single.shape[0]):
    #    if p_values_single[i] <= 0.05:
    #        ax.text(i + 0.5, 0.5, '*', ha='center', va='center', color='black', fontsize=10)

    for i in range(df_p_values_single_sorted.shape[0]):
        if (df_p_values_single_sorted.iloc[i] <= 0.05).any():
            ax.text(i + 0.5, 0.5, '*', ha='center', va='center', color='black', fontsize=10)

    # Rotate x-axis ticks
    ax.tick_params(axis='x', rotation=45, labelsize=5)

    # Set x-ticks and corresponding labels for each volume
    ax.set_xticks(np.arange(len(df_corr_single)))
    ax.set_xticklabels(volume_numbers)  # Set volume numbers as xticklabels

    # Clear volume_numbers for the next brain map
    volume_numbers = []

# Remove empty subplots
for i in range(num_maps, num_cols * num_rows):
    fig.delaxes(axes.flatten()[i])

# Adjust layout to prevent overlapping and increase space between rows
plt.tight_layout(pad=12.0)  # 3.0

# Save the plot as an image
plt.savefig('heatmap_combined.png')

# Show the plot
plt.show()
'''


# Calculate the number of rows and columns for the subplot grid
num_maps = len(annotation_sources)
num_cols = 3  # Number of columns in the grid
num_rows = (num_maps + num_cols - 1) // num_cols  # Calculate the number of rows needed

# Create a new figure with the appropriate size
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8 * num_rows))

corr_values_all = []

# Loop through each brain map and corresponding axis in the grid
for idx, source in enumerate(annotation_sources):
    row_idx = idx // num_cols
    col_idx = idx % num_cols
    ax = axes[row_idx, col_idx]  # Get the axis corresponding to the current brain map

    # Fetch annotation
    anno = fetch_annotation(source=source)
    # Initialize correlation values list
    corr_values_single = []
    p_values_single = []
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

        volume_numbers.append((i, corr))  # Store the volume number and its corresponding correlation value

        print(f"Processing {volume_path}")
        print(f'r = {corr:.3f}')
        print("\n")

    corr_values_all.extend(corr_values_single)
    # Calculate the highest and lowest total correlation values
    min_corr = min(corr_values_all)
    max_corr = max(corr_values_all)

    # Create a DataFrame for correlation values for the current brain map
    df_corr_single = pd.DataFrame({'Correlation Values': corr_values_single})

    df_p_values_single = pd.DataFrame({'p-Values': p_values_single})

    # Sort correlation values from lowest to highest
    df_corr_single_sorted = df_corr_single.sort_values(by='Correlation Values')

    # Sort p-values based on the sorted correlation values
    df_p_values_single_sorted = df_p_values_single.loc[df_corr_single_sorted.index]

    # Plot the heatmap of sorted correlation values for the current brain map
    sns.heatmap(df_corr_single_sorted.transpose(), annot=False, fmt='.2f', cmap='coolwarm', cbar=True, center=(min_corr + max_corr) / 2,
                ax=ax, vmin=min_corr, vmax=max_corr)
    ax.set_title(f'{source}')
    ax.set_xlabel('Volume Number')
    if col_idx == 0:  # Add y-axis label only for the first column
        ax.set_ylabel('Correlation Value')
    else:
        ax.set_ylabel('')  # Remove y-axis label for other columns

    # Loop through each cell and add the p-value as annotation text if it's significant
    #for i in range(df_corr_single.shape[0]):
    #    if p_values_single[i] <= 0.05:
    #        ax.text(i + 0.5, 0.5, '*', ha='center', va='center', color='black', fontsize=10)

    for i in range(df_p_values_single_sorted.shape[0]):
        if (df_p_values_single_sorted.iloc[i] <= 0.05).any():
            ax.text(i + 0.5, 0.5, '*', ha='center', va='center', color='black', fontsize=10)

    # Rotate x-axis ticks
    ax.tick_params(axis='x', rotation=45, labelsize=6)

    # Sort volume numbers based on the correlation values
    volume_numbers_sorted = sorted(volume_numbers, key=lambda x: x[1])

    # Extract only the sorted volume numbers
    sorted_volume_numbers = [volume[0] for volume in volume_numbers_sorted]

    # Set x-ticks and corresponding labels for each volume
    ax.set_xticks(np.arange(len(df_corr_single)))
    ax.set_xticklabels(sorted_volume_numbers[:len(df_corr_single)])  # Set sorted volume numbers as xticklabels

    # Extend the list with correlation values for the current brain map
    corr_values_all.extend(corr_values_single)

# Remove empty subplots
for i in range(num_maps, num_cols * num_rows):
    fig.delaxes(axes.flatten()[i])

# Adjust layout to prevent overlapping and increase space between rows
plt.tight_layout(pad=12.0)  # 3.0

plt.suptitle('Bootstrapped correlations of tVNS-induced changes in rs-FC and receptor maps (single subject level)', fontsize=20, fontweight='bold', y=1.1)
# Save the plot as an image
plt.savefig('heatmap_combined.png')

# Show the plot
plt.show()



