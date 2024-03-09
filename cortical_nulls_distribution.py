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
import arviz as az
import xarray as xr
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from nilearn import image as nli
import matplotlib.pyplot as plt
from netneurotools.datasets import fetch_schaefer2018
from neuromaps import transforms, stats
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
print(mean_img_fslr)
# The output is a tuple of Gifti Images

# Parcellate my image
mean_img_fslr_parc = parc_fsLR.fit_transform(mean_img_fslr, 'fsLR')
# The output is an array


#-------- NULLS FOR MEAN IMAGE --------#

# Generate nulls
# Only once before the loop
nulls_mean = alexander_bloch(mean_img_fslr_parc, atlas='fsLR', density='32k', parcellation=parcels_fslr_32k)
print(f'These are the Nulls for the Mean Image: {nulls_mean}')
print(len(nulls_mean))
# Should be a ndarray with shape 400(, 1000)


#-------- SPATIAL CORRELATIONS OF THE NULLS FOR THE MEAN IMAGE --------#

# Fetch annotation
ding2010 = fetch_annotation(source='ding2010')
# Ding 2010 is in MNI space with 1mm density
# Transformation to the fsLR space (same density as your transformed data: 32k)
ding2010_fslr = transforms.mni152_to_fslr(ding2010, '32k')
# The annotation and your data is now both fsLR 32k
# Parcellate annotation
ding2010_fslr_parc = parc_fsLR.fit_transform(ding2010_fslr, 'fsLR')


# Calculate spatial correlation and p-value
# WRONG COMMAND
#corr_mean, pval_mean = stats.compare_images(mean_img_fslr_parc, ding2010_fslr_parc, nulls=nulls_mean)
#print(f"Correlation for neuromaps annotation and mean image: {corr_mean[0]}")
#print(f"p-value for annotation and mean image: {corr_mean[1]}")
#print(f'Correlation value for annotation and mean image: {corr_mean}')
#print(f'P-value for annotation and mean image: {pval_mean}')

# CORRECT COMMAND
# pval as well?
#corr_nulls_mean = stats.compare_images(mean_img_fslr_parc, ding2010_fslr_parc, metric='pearsonr', ignore_zero=True, nulls=nulls_mean, nan_policy='omit', return_nulls=True)
#print(f'Here are the Nulls for the Mean Image with the Annotation:{corr_nulls_mean}')
#print(corr_nulls_mean)


#-------- Histogram plotting -------#

# TEST
# Create a histogram for the current map
#plt.hist(corr_nulls_mean, edgecolor='black')#ax.axvline(q5, color='black', linestyle='dashed', linewidth=2, label='5% Quantile')
#ax.axvline(q95, color='black', linestyle='dashed', linewidth=2, label='95% Quantile')
#plt.show()



################## CORRECTED VERSION ####################

# Mapping of annotation sources to categories
category_mapping = {
    'ding2010': 'NE',
    'hesse2017': 'NE',
    'kaller2017': 'D1',
    'alarkurtti2015': 'D2/D3',
    'jaworska2020': 'D2/D3',
    'sandiego2015': 'D2/D3',
    'smith2017': 'D2/D3',
    'sasaki2012': 'DAT',
    'fazio2016': '5-HTT',
    'gallezot2010': '5-HTb',
    'radnakrishnan2018': '5-HT6'
}

#--------- LOOP FOR SC FOR MEAN IMAGE WITH 11 MAPS ---------#

# List of annotation sources
#annotation_sources = ['alarkurtti2015', 'ding2010', 'fazio2016', 'gallezot2010', 'hesse2017',
#                      'jaworska2020', 'kaller2017', 'radnakrishnan2018', 'sandiego2015', 'sasaki2012', 'smith2017']
annotation_sources = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

# Manually define colors for each histogram
hist_colors = ['blue', 'blue', 'olive', 'green', 'green', 'green', 'green', 'cyan', 'red', 'orange', 'yellow']

# Create list for saving the r-value, the p-value and the nulls for each correlation
corr_vals_mean_list = []
pval_mean_list = []
corr_nulls_mean_list = []

# Define the number of columns you want
num_columns = 2

# Calculate the number of rows based on the number of annotation sources and the number of columns
num_rows = -(-len(annotation_sources) // num_columns)  # Ceiling division to ensure all items are shown

# All histograms in one figure
#fig, axs = plt.subplots(len(annotation_sources), figsize=(12, 6), sharex=False, sharey=False)
fig, axs = plt.subplots(num_rows, num_columns, figsize=(12, 12), sharex=False, sharey=False)

#colors = [(0, 0, 0.5, 0.1), (0, 0, 0.5, 0.5), (0, 0, 0.5, 0.9)]
#cmap_name = 'blues'
#cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=10)

for i, source in enumerate(annotation_sources):
    row = i // num_columns
    col = i % num_columns

    # Fetch annotation
    annotation = fetch_annotation(source=source)

    # Transformation to the fsLR space (sam density as your transformed data: 32k)
    annotation_fslr = transforms.mni152_to_fslr(annotation, '32k')
    # The annotation and your data is now both fsLR 32k

    # Parcellate annotation
    annotation_fslr_parc = parc_fsLR.fit_transform(annotation_fslr, 'fsLR')

    # Calculate spatial correlation
    corr_val_mean, pval_mean, corr_nulls_mean = stats.compare_images(mean_img_fslr_parc, annotation_fslr_parc, metric='pearsonr', ignore_zero=True,
                                           nulls=nulls_mean, nan_policy='omit', return_nulls=True)

    # Append to lists
    corr_vals_mean_list.append(corr_val_mean)
    pval_mean_list.append(pval_mean)
    corr_nulls_mean_list.append(corr_nulls_mean)

    #print("r-Value:", corr_val_mean)
    #print("p-Value:", pval_mean)
    #print("Nulls:", corr_nulls_mean)

    #print(f'Correlation value for annotation ({source}) and nulls of the mean image: {corr_nulls_mean}')

    # Determine the correct subplot to use
    if num_rows == 1:
        subplot_ax = axs[col]
    else:
        subplot_ax = axs[row, col]

    # Create a histogram for the current map
    sns.histplot(data=corr_nulls_mean, color=hist_colors[i], edgecolor='black', kde=True,
                 ax=subplot_ax, legend=False)
    # color=cm(i / len(annotation_sources))

   # Calculate 95% confidence interval for each array in corr_nulls_mean
#    for arr in corr_nulls_mean:
#        q5, q95 = np.percentile(np.array(arr).flatten(), [5, 95])
#        subplot_ax.axvline(q5, color='black', linestyle='dashed', linewidth=2, label='5% Quantile')
#        subplot_ax.axvline(q95, color='black', linestyle='dashed', linewidth=2, label='95% Quantile')

    # Plot 95% confidence interval
    #subplot_ax.axvline(q5, color='black', linestyle='dashed', linewidth=2, label='5%')
    #subplot_ax.axvline(q95, color='black', linestyle='dashed', linewidth=2, label='95%')
    #subplot_ax.axvline(q5, color='black', linestyle='dashed', linewidth=2, label='5% Quantile')
    #subplot_ax.axvline(q95, color='black', linestyle='dashed', linewidth=2, label='95% Quantile')

    # Plot vertical line for the similarity value
    subplot_ax.axvline(corr_val_mean, color='red', linestyle='dashed', linewidth=2, label='Original r-Value')

    # Add p-value as text annotation in the top right part of each histogram
    subplot_ax.text(0.95, 0.95, f'p = {pval_mean:.3f}', transform=subplot_ax.transAxes, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.5))

    subplot_ax.set_xlabel('Spatial Nulls')
    subplot_ax.set_ylabel('Frequency')
    subplot_ax.set_title(source)

# Remove empty subplots if the number of sources is not a multiple of the number of columns
for i in range(len(annotation_sources), num_rows * num_columns):
    fig.delaxes(axs.flatten()[i])

# Set the limits of the x and y axes manually
#axs[i].axis([0, 1, 0, 1])
plt.subplots_adjust(wspace=0.3, hspace=3) # adds twice the default space between the plots

# Add custom legend for the colors
unique_colors = list(set(hist_colors))  # Get unique colors
legend_labels = ['NE', 'D1', 'D2/3', 'DAT', '5-HTT', '5-HTb', '5-HT6']  # Corresponding labels for unique colors

# Create legend handles and labels
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label)
                  for color, label in zip(unique_colors, legend_labels)]

# Add legend with handles and labels
#fig.legend(handles=legend_handles, labels=legend_labels, loc='upper right')


# Add custom legend for the similarity value
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
#plt.tight_layout()
plt.show()

# color legend
# confidence interval?
##############################################################################



'''
#--------- SPATIAL CORRELATIONS FOR MEAN IMAGE --------#

# Fetch annotation
ding2010 = fetch_annotation(source='ding2010')
# Ding 2010 is in MNI space with 1mm density

# Transformation to the fsLR space (same density as your transformed data: 32k)
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
#annotation_sources = ['alarkurtti2015', 'ding2010', 'fazio2016', 'gallezot2010', 'hesse2017',
#                      'jaworska2020', 'kaller2017', 'radnakrishnan2018', 'sandiego2015', 'sasaki2012', 'smith2017']
annotation_sources = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

corr_values_mean = []
p_values_mean = []
data = []

# THIS LOOP IS COMMENTED OUT
# One figure for each histogram (i.e. each map)
for source in annotation_sources:
    # Fetch annotation
    annotation = fetch_annotation(source=source)

    # Transformation to the fsLR space (sam density as your transformed data: 32k)
    annotation_fslr = transforms.mni152_to_fslr(annotation, '32k')
    # The annotation and your data is now both fsLR 32k

    # Parcellate annotation
    annotation_fslr_parc = parc_fsLR.fit_transform(annotation_fslr, 'fsLR')

    # Calculate null distribution
    nulls_annotation = alexander_bloch(annotation_fslr_parc, atlas='fsLR', density='32k', parcellation=parcels_fslr_32k)

    # 2) Calculate spatial correlation and p-value
    corr_mean, pval_mean = stats.compare_images(mean_img_fslr_parc, annotation_fslr_parc, nulls=nulls_mean)
    print(f'Correlation value for annotation ({source}) and mean image: {corr_mean}')
    print(f'P-value for annotation ({source}) and mean image: {pval_mean}')

    corr_values_mean.append(corr_mean)
    p_values_mean.append(pval_mean)

    # Flatten the null distribution and add it to the data list
    data.append(np.array(nulls_annotation).flatten())

    # Calculate 5% quantiles
    q5, q95 = np.percentile(np.array(nulls_annotation).flatten(), [5, 95])

    # Histogram Plotting
    plt.figure(figsize=(12, 6))
    plt.hist(np.array(nulls_annotation).flatten(), bins=30, color='blue',
             alpha=0.7, edgecolor='black', stacked=True)
    # Add line for p-value?
    plt.axvline(q5, color='red', linestyle='dashed', linewidth=2, label='5% Quantile')
    plt.axvline(q95, color='green', linestyle='dashed', linewidth=2, label='95% Quantile')
    plt.xlabel('Spatial Correlation')
    plt.ylabel('Density')
    plt.title(f'Histogram of Null Distribution of the Mean Image with {source}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(corr_values_mean)
    print(pval_mean)

# COMMENTED OUT UNTIL HERE

# Define the number of columns you want
num_columns = 2

# Calculate the number of rows based on the number of annotation sources and the number of columns
num_rows = -(-len(annotation_sources) // num_columns)  # Ceiling division to ensure all items are shown

# All histograms in one figure
#fig, axs = plt.subplots(len(annotation_sources), figsize=(12, 6), sharex=False, sharey=False)
fig, axs = plt.subplots(num_rows, num_columns, figsize=(12, 12), sharex=False, sharey=False)

colors = [(0, 0, 0.5, 0.1), (0, 0, 0.5, 0.5), (0, 0, 0.5, 0.9)]
cmap_name = 'blues'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=10)

for i, source in enumerate(annotation_sources):
    row = i // num_columns
    col = i % num_columns

    # Fetch annotation
    annotation = fetch_annotation(source=source)

    # Transformation to the fsLR space (sam density as your transformed data: 32k)
    annotation_fslr = transforms.mni152_to_fslr(annotation, '32k')
    # The annotation and your data is now both fsLR 32k

    # Parcellate annotation
    annotation_fslr_parc = parc_fsLR.fit_transform(annotation_fslr, 'fsLR')

    # Calculate null distribution
    nulls_annotation = alexander_bloch(annotation_fslr_parc, atlas='fsLR', density='32k', parcellation=parcels_fslr_32k)

    # 2) Calculate spatial correlation and p-value
    corr_mean, pval_mean = stats.compare_images(mean_img_fslr_parc, annotation_fslr_parc, nulls=nulls_mean)
    print(f'Correlation value for annotation ({source}) and mean image: {corr_mean}')
    print(f'P-value for annotation ({source}) and mean image: {pval_mean}')

    # Flatten the null distribution and add it to the data list
    data = np.array(nulls_annotation).flatten()
    # THIS LINE MIGHT BE THE ERROR

    # Calculate 5% quantiles
    q5, q95 = np.percentile(data, [5, 95])
    # Determine the correct subplot to use
    if num_rows == 1:
        subplot_ax = axs[col]
    else:
        subplot_ax = axs[row, col]

    # Create a histogram for the current map
    sns.histplot(data=data, bins=30, color=cm(i / len(annotation_sources)), edgecolor='black', kde=True, ax=subplot_ax)
    subplot_ax.axvline(q5, color='black', linestyle='dashed', linewidth=2, label='5% Quantile')
    subplot_ax.axvline(q95, color='black', linestyle='dashed', linewidth=2, label='95% Quantile')
    subplot_ax.set_xlabel('Spatial Correlation')
    subplot_ax.set_ylabel('Frequency')
    #subplot_ax.set_title(source)

    # Add p-value as text annotation
    #axs[row, col].text(0.5, 0.5, f'p-value: {pval_mean:.3f}', horizontalalignment='center', verticalalignment='bottom',
    #                   transform=axs[row, col].transAxes)
    subplot_ax.set_title(f'{source}\n(p-value: {pval_mean:.3f})', loc='center', fontsize=10)

# Remove empty subplots if the number of sources is not a multiple of the number of columns
for i in range(len(annotation_sources), num_rows * num_columns):
    fig.delaxes(axs.flatten()[i])


# Set the limits of the x and y axes manually
#axs[i].axis([0, 1, 0, 1])
plt.subplots_adjust(wspace=0.3, hspace=3) # adds twice the default space between the plots
#plt.tight_layout()
plt.show()

# something with the axes and values is off
# incorrect calculation? i don't think so

'''



