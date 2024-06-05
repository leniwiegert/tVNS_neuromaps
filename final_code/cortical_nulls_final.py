'''
@author: Lena Wiegert

This code transforms MNI152 data to fsLR space and parcellates it with the Schaefer atlas.
It further visualizes the significance of the spatial correlations on group level for cortical data.
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


#-------- NULLS FOR group IMAGE --------#

# Generate nulls
nulls_group = alexander_bloch(group_img_fslr_parc, atlas='fsLR', density='32k', parcellation=parcels_fslr_32k, n_perm=1000)
print(f'These are the Nulls for the group Image: {nulls_group}')
print(len(nulls_group))
# Should be a ndarray with shape 400(, 1000)


#-------- SPATIAL NULLS ON GROUP LEVEL --------#

# Fetch annotation
ding2010 = fetch_annotation(source='ding2010')
# Ding 2010 is in MNI space with 1mm density
# Transformation to the fsLR space (same density as the transformed data: 32k)
ding2010_fslr = transforms.mni152_to_fslr(ding2010, '32k')
# The annotation and the data is now both fsLR 32k
# Parcellate annotation
ding2010_fslr_parc = parc_fsLR.fit_transform(ding2010_fslr, 'fsLR')

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

# List of annotation sources
annotation_sources = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

# Manually define colors for each histogram
hist_colors = ['blue', 'blue', 'olive', 'green', 'green', 'green', 'green', 'cyan', 'red', 'orange', 'yellow']

# Create list for saving the r-value, the p-value and the nulls for each correlation
corr_vals_group_list = []
pval_group_list = []
corr_nulls_group_list = []

# Define the number of columns you want
num_columns = 2

# Calculate the number of rows based on the number of annotation sources and the number of columns
num_rows = -(-len(annotation_sources) // num_columns)  # Ceiling division to ensure all items are shown

# All histograms in one figure
fig, axs = plt.subplots(num_rows, num_columns, figsize=(12, 12), sharex=False, sharey=False)

for i, source in enumerate(annotation_sources):
    row = i // num_columns
    col = i % num_columns

    # Fetch annotation
    annotation = fetch_annotation(source=source)

    # Transformation to the fsLR space (samE density as the transformed data: 32k)
    annotation_fslr = transforms.mni152_to_fslr(annotation, '32k')
    # The annotation and the data is now both fsLR 32k

    # Parcellate annotation
    annotation_fslr_parc = parc_fsLR.fit_transform(annotation_fslr, 'fsLR')

    # Calculate spatial correlation
    corr_val_group, pval_group, corr_nulls_group = stats.compare_images(group_img_fslr_parc, annotation_fslr_parc, metric='pearsonr', ignore_zero=True,
                                           nulls=nulls_group, nan_policy='omit', return_nulls=True)

    # Append to lists
    corr_vals_group_list.append(corr_val_group)
    pval_group_list.append(pval_group)
    corr_nulls_group_list.append(corr_nulls_group)

    print(f'Correlation value for annotation ({source}) and nulls of the group image: {corr_nulls_group}')

    # Determine the correct subplot to use
    if num_rows == 1:
        subplot_ax = axs[col]
    else:
        subplot_ax = axs[row, col]

    # Create a histogram for the current map
    sns.histplot(data=corr_nulls_group, color=hist_colors[i], edgecolor='black', kde=True,
                 ax=subplot_ax, legend=False)

    # Plot vertical line for the similarity value
    subplot_ax.axvline(corr_val_group, color='red', linestyle='dashed', linewidth=2, label='Original r-Value')

    # Add p-value as text annotation in the top right part of each histogram
    subplot_ax.text(0.95, 0.95, f'p = {pval_group:.3f}', transform=subplot_ax.transAxes, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.5))

    subplot_ax.set_xlabel('Spatial Nulls', fontsize=16)
    subplot_ax.set_ylabel('Frequency', fontsize=16)
    subplot_ax.set_title(source, fontsize=16)

# Remove empty subplots if the number of sources is not a multiple of the number of columns
for i in range(len(annotation_sources), num_rows * num_columns):
    fig.delaxes(axs.flatten()[i])
plt.subplots_adjust(wspace=0.3, hspace=3)

# Add custom legend for the colors
unique_colors = list(set(hist_colors))  # Get unique colors
legend_labels = ['NE', 'D1', 'D2/3', 'DAT', '5-HTT', '5-HTb', '5-HT6']  # Corresponding labels for unique colors

# Create legend handles and labels
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label)
                  for color, label in zip(unique_colors, legend_labels)]

# Add custom legend for the similarity value
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.suptitle('Correlations cortical maps of tVNS-induced changes and receptor maps on group level', fontsize=20)
plt.show()


#------------- SAVE FIGURE ------------#

# Define the filename for the figure
figure_filename = 'cortical_nulls_final.png'
# Construct the full file path
figure_path = os.path.join(data_directory, figure_filename)
# Save the figure
fig.savefig(figure_path, dpi=300, bbox_inches='tight')

