'''
@author: Lena Wiegert

This code compares the spatial correlations of parcellated cortical maps of tVNS-induced changes
to the same data as parcellated subcortical maps. These steps are performed on group level.
Further, an analysis based on the 7 subnetworks included in the Schaefer Atlas is performed.
'''

import os
import numpy as np
import nibabel as nib
import nilearn as nli
from neuromaps.images import dlabel_to_gifti, relabel_gifti
from neuromaps.nulls import alexander_bloch
from neuromaps.parcellate import Parcellater
import matplotlib.pyplot as plt
from neuromaps import transforms, stats
from neuromaps.datasets import fetch_annotation
from netneurotools.datasets import fetch_schaefer2018


#-------- LOAD AND PREP DATA --------#

# Define universal data directory
data_directory = '/home/leni/Documents/Master/data/'
#data_directory = '/home/neuromadlab/tVNS_project/data/'

img = nib.load(os.path.join(data_directory, 'parcellated_volume_37.nii'))
img_data = img.get_fdata()
# Replace non-finite values with a gray matter mask
gray_matter_mask_file = os.path.join(data_directory, 'out_GM_p_0_15.nii')
gray_matter_mask = nib.load(gray_matter_mask_file)
#vol_1_gm = os.path.join(data_directory, 'vol_1_gm.nii')
mean_img_gm = os.path.join(data_directory, 'mean_img_gm.nii')


#-------- CORTICAL AND SUBCORTICAL PARCELLATION --------#

# Cortical parcellation
# Import the parcellation maps (Schaefer) in fsLR space
parcels_fslr_32k = fetch_schaefer2018('fslr32k')['400Parcels7Networks']
parcels_fslr_32k = dlabel_to_gifti(parcels_fslr_32k)
parcels_fslr_32k = relabel_gifti(parcels_fslr_32k)
mean_img_fslr = transforms.mni152_to_fslr(mean_img_gm, '32k')

# Create parcellaters for fsLR
parc_fsLR = Parcellater(parcels_fslr_32k, 'fslr', resampling_target=None)
mean_img_fslr_parc = parc_fsLR.fit_transform(mean_img_fslr, 'fsLR')
# Create null model for individual data
nulls_individual = alexander_bloch(mean_img_fslr_parc, atlas='fsLR', density='32k', parcellation=parcels_fslr_32k)
print(len(nulls_individual))

# Subcortical parcellation
atlas_path = '/home/leni/Tian2020MSA_v1.4/Tian2020MSA/3T/Subcortex-Only/Tian_Subcortex_S4_3T.nii'
atlas_img = nib.load(atlas_path)
parcellater = Parcellater(parcellation=atlas_img,  space='MNI152')
parcellater.fit()

# Load all volumes
volumes = []
for i in range(1,42):
    filename = os.path.join(data_directory, f'volume_{i}.nii')
    # Load the NIfTI file
    img = nib.load(filename)
    # Get the image data as a NumPy array
    data = img.get_fdata()
    # Append the data to the list
    volumes.append(data)

# Convert the list of volumes to a NumPy array
volumes_array = np.array(volumes)

# Parcellation for all volumes
annotation_sources = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

for source in annotation_sources:
    all_corr_values_individual = []

    # Fetch annotation
    anno = fetch_annotation(source=source)

    # Process each volume
    for i in range(1, 42):
        # Initialize correlation values list for each volume
        corr_values_individual = []

        volume_path = os.path.join(data_directory, f'volume_{i}.nii')
        volume = nib.load(volume_path)
        # Resample and add GM mask
        gray_matter_mask_resampled = nli.image.resample_to_img(gray_matter_mask, volume)
        # Create a new mask by keeping only non-NaN values in both masks
        vol_gm_data = np.where(np.isnan(volume.get_fdata()), gray_matter_mask_resampled.get_fdata(), volume.get_fdata())
        # Create a new image with the new mask
        vol_gm = nib.Nifti1Image(vol_gm_data.astype(np.float32), volume.affine)

        # Parcellate individual volumes
        vol_subcort = parcellater.transform(vol_gm, space='MNI152')

        # Save the parcellated volume as a NumPy array
        parc_vol_path = os.path.join(data_directory, f'parc_subcort_vol_{i}.npy')
        np.save(parc_vol_path, vol_subcort)
        print(f"Processed and saved {parc_vol_path}")

    # Append all correlation values for the current annotation source to a master list
    # You can use this list for further analysis if needed
    #all_corr_values_individual.append(corr_values_individual)


#-------- ROI-BASED COMPARISON OF CORTICAL AND SUBCORTICAL DATA --------#

# Load the brain atlas defining the ROIs
mean_img_fslr_parc = mean_img_fslr_parc

# Load the brain atlas defining the ROIs
mean_img_subcort_parc = parcellater.transform(mean_img_gm, space='MNI152')

# Parcellate the annotations as well:
des_anno = fetch_annotation(source='ding2010')
# Transform annotation to fsLR
des_anno_fslr = transforms.mni152_to_fslr(des_anno, '32k')
# Parcellate annotation
des_anno_fslr_parc = parc_fsLR.fit_transform(des_anno_fslr, 'fsLR')
des_anno_mni152_parc = parcellater.transform(des_anno, space='MNI152')

des_anno_2 = fetch_annotation(source='hesse2017')
# Transform annotation to fsLR
des_anno_fslr_2 = transforms.mni152_to_fslr(des_anno_2, '32k')
# Parcellate annotation
des_anno_fslr_parc_2 = parc_fsLR.fit_transform(des_anno_fslr_2, 'fsLR')
des_anno_mni152_parc_2 = parcellater.transform(des_anno_2, space='MNI152')
print('Hesse2017 Debug:')
print(des_anno_fslr_parc_2)
print(des_anno_mni152_parc_2)

des_anno_3 = fetch_annotation(source='kaller2017')
# Transform annotation to fsLR
des_anno_fslr_3 = transforms.mni152_to_fslr(des_anno_3, '32k')
# Parcellate annotation
des_anno_fslr_parc_3 = parc_fsLR.fit_transform(des_anno_fslr_3, 'fsLR')
des_anno_mni152_parc_3 = parcellater.transform(des_anno_3, space='MNI152')

des_anno_4 = fetch_annotation(source='alarkurtti2015')
# Transform annotation to fsLR
des_anno_fslr_4 = transforms.mni152_to_fslr(des_anno_4, '32k')
# Parcellate annotation
des_anno_fslr_parc_4 = parc_fsLR.fit_transform(des_anno_fslr_4, 'fsLR')
des_anno_mni152_parc_4 = parcellater.transform(des_anno_4, space='MNI152')

des_anno_5 = fetch_annotation(source='jaworska2020')
# Transform annotation to fsLR
des_anno_fslr_5 = transforms.mni152_to_fslr(des_anno_5, '32k')
# Parcellate annotation
des_anno_fslr_parc_5 = parc_fsLR.fit_transform(des_anno_fslr_5, 'fsLR')
des_anno_mni152_parc_5 = parcellater.transform(des_anno_5, space='MNI152')

des_anno_6 = fetch_annotation(source='sandiego2015')
# Transform annotation to fsLR
des_anno_fslr_6 = transforms.mni152_to_fslr(des_anno_6, '32k')
# Parcellate annotation
des_anno_fslr_parc_6 = parc_fsLR.fit_transform(des_anno_fslr_6, 'fsLR')
des_anno_mni152_parc_6 = parcellater.transform(des_anno_6, space='MNI152')

des_anno_7 = fetch_annotation(source='smith2017')
# Transform annotation to fsLR
des_anno_fslr_7 = transforms.mni152_to_fslr(des_anno_7, '32k')
# Parcellate annotation
des_anno_fslr_parc_7 = parc_fsLR.fit_transform(des_anno_fslr_7, 'fsLR')
des_anno_mni152_parc_7 = parcellater.transform(des_anno_7, space='MNI152')

des_anno_8 = fetch_annotation(source='sasaki2012')
# Transform annotation to fsLR
des_anno_fslr_8 = transforms.mni152_to_fslr(des_anno_8, '32k')
# Parcellate annotation
des_anno_fslr_parc_8 = parc_fsLR.fit_transform(des_anno_fslr_8, 'fsLR')
des_anno_mni152_parc_8 = parcellater.transform(des_anno_8, space='MNI152')

des_anno_9 = fetch_annotation(source='fazio2016')
# Transform annotation to fsLR
des_anno_fslr_9 = transforms.mni152_to_fslr(des_anno_9, '32k')
# Parcellate annotation
des_anno_fslr_parc_9 = parc_fsLR.fit_transform(des_anno_fslr_9, 'fsLR')
des_anno_mni152_parc_9 = parcellater.transform(des_anno_9, space='MNI152')

des_anno_10 = fetch_annotation(source='gallezot2010')
# Transform annotation to fsLR
des_anno_fslr_10 = transforms.mni152_to_fslr(des_anno_10, '32k')
# Parcellate annotation
des_anno_fslr_parc_10 = parc_fsLR.fit_transform(des_anno_fslr_10, 'fsLR')
des_anno_mni152_parc_10 = parcellater.transform(des_anno_10, space='MNI152')

des_anno_11 = fetch_annotation(source='radnakrishnan2018')
# Transform annotation to fsLR
des_anno_fslr_11 = transforms.mni152_to_fslr(des_anno_11, '32k')
# Parcellate annotation
des_anno_fslr_parc_11 = parc_fsLR.fit_transform(des_anno_fslr_11, 'fsLR')
des_anno_mni152_parc_11 = parcellater.transform(des_anno_11, space='MNI152')

# Plotting:
# In this example, the relation of tVNS-induced changes with dopamine receptor maps is shown.
# Change the number of des_anno_parc accordingly to the map you want to test.

fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Plot for Alarkurtti2015
axs[0, 0].scatter(mean_img_fslr_parc, des_anno_fslr_parc_4, label='Cortical', s=30, alpha=0.7, color='darkred')
axs[0, 0].scatter(mean_img_subcort_parc, des_anno_mni152_parc_4, label='Subcortical', s=30, alpha=0.7, color='orange')
axs[0, 0].set_title('Alarkurtti2015', fontsize=14)
axs[0, 0].set_xlabel('Maps of tVNS-induced Changes', fontsize=12)
axs[0, 0].set_ylabel('Receptor Density Map', fontsize=12)

# Plot for Jaworska2020
axs[0, 1].scatter(mean_img_fslr_parc, des_anno_fslr_parc_5, label='Cortical', s=30, alpha=0.7, color='darkred')
axs[0, 1].scatter(mean_img_subcort_parc, des_anno_mni152_parc_5, label='Subcortical', s=30, alpha=0.7, color='orange')
axs[0, 1].set_title('Jaworska2020', fontsize=14)
axs[0, 1].set_xlabel('Maps of tVNS-induced Changes', fontsize=12)
axs[0, 1].set_ylabel('Receptor Density Map', fontsize=12)

# Plot for Sandiego2015
axs[1, 0].scatter(mean_img_fslr_parc, des_anno_fslr_parc_6, label='Cortical', s=30, alpha=0.7, color='darkred')
axs[1, 0].scatter(mean_img_subcort_parc, des_anno_mni152_parc_6, label='Subcortical', s=30, alpha=0.7, color='orange')
axs[1, 0].set_title('Sandiego2015', fontsize=14)
axs[1, 0].set_xlabel('Maps of tVNS-induced Changes', fontsize=12)
axs[1, 0].set_ylabel('Receptor Density Map', fontsize=12)

# Plot for Smith2017
axs[1, 1].scatter(mean_img_fslr_parc, des_anno_fslr_parc_7, label='Cortical', s=30, alpha=0.7, color='darkred')
axs[1, 1].scatter(mean_img_subcort_parc, des_anno_mni152_parc_7, label='Subcortical', s=30, alpha=0.7, color='orange')
axs[1, 1].set_title('Smith2017', fontsize=14)
axs[1, 1].set_xlabel('Maps of tVNS-induced Changes', fontsize=12)
axs[1, 1].set_ylabel('Receptor Density Map', fontsize=12)

# Optional: Customize spines
for ax in axs.flat:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('grey')
        spine.set_linewidth(0.5)

# Setting common title
fig.suptitle('Correlation of Cortical and Subcortical Maps of tVNS-induced changes with D2/3 Receptor Maps', fontsize=16)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Save the figure
fig_path = os.path.join(data_directory, 'cortical_subcortical_roi_dopamine.png')
plt.savefig(fig_path)
plt.show()


#-------- SUBNETWORK ANALYSIS --------#

# Define RGB colors for each subnetwork (example)
subnetwork_colors = {
    'Visual': ('Purple'),
    'Somatomotor': ('Blue'),
    'Dorsal Attention': ('Green'),
    'Ventral Attention': ('Violet'),
    'Limbic': ('Grey'),
    'Frontoparietal': ('Orange'),
    'Default': ('Red')
}

# Function to map ROI index to subnetwork color
def map_roi_to_color(roi_index):
    if 1 <= roi_index <= 31:  # Visual Network (Left Hemisphere)
        return subnetwork_colors['Visual']
    elif 32 <= roi_index <= 68:  # Somatomotor Network (Left Hemisphere)
        return subnetwork_colors['Somatomotor']
    elif 69 <= roi_index <= 91:  # Dorsal Attention Network (Left Hemisphere)
        return subnetwork_colors['Dorsal Attention']
    elif 92 <= roi_index <= 113:  # Ventral Attention Network (Left Hemisphere)
        return subnetwork_colors['Ventral Attention']
    elif 114 <= roi_index <= 126:  # Limbic Network (Left Hemisphere)
        return subnetwork_colors['Limbic']
    elif 127 <= roi_index <= 148:  # Frontoparietal Network (Left Hemisphere)
        return subnetwork_colors['Frontoparietal']
    elif 149 <= roi_index <= 200:  # Default Network (Left Hemisphere)
        return subnetwork_colors['Default']
    elif 201 <= roi_index <= 230:  # Visual Network (Right Hemisphere)
        return subnetwork_colors['Visual']
    elif 231 <= roi_index <= 270:  # Somatomotor Network (Right Hemisphere)
        return subnetwork_colors['Somatomotor']
    elif 271 <= roi_index <= 293:  # Dorsal Attention Network (Right Hemisphere)
        return subnetwork_colors['Dorsal Attention']
    elif 294 <= roi_index <= 318:  # Ventral Attention Network (Right Hemisphere)
        return subnetwork_colors['Ventral Attention']
    elif 319 <= roi_index <= 331:  # Limbic Network (Right Hemisphere)
        return subnetwork_colors['Limbic']
    elif 332 <= roi_index <= 361:  # Frontoparietal Network (Right Hemisphere)
        return subnetwork_colors['Frontoparietal']
    elif 362 <= roi_index <= 400:  # Default Network (Right Hemisphere)
        return subnetwork_colors['Default']

# Plotting with corresponding colors

fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Plot for Alarkurtti2015
axs[0, 0].set_title('Alarkurtti2015', fontsize=14)
for roi_index in range(len(mean_img_fslr_parc)):
    color = map_roi_to_color(roi_index)  # Map ROI to corresponding color
    axs[0, 0].scatter(mean_img_fslr_parc[roi_index], des_anno_fslr_parc_4[roi_index],
                label='Cortical', s=50, alpha=0.7, color=color)

axs[0, 0].set_xlabel('Cortical Map of tVNS-induced Changes', fontsize=12)
axs[0, 0].set_ylabel('Receptor Density Map', fontsize=12)

# Plot for Jaworska2020
axs[0, 1].set_title('Jaworska2020', fontsize=14)
for roi_index in range(len(mean_img_fslr_parc)):
    color = map_roi_to_color(roi_index)  # Map ROI to corresponding color
    axs[0, 1].scatter(mean_img_fslr_parc[roi_index], des_anno_fslr_parc_5[roi_index],
                label='Cortical', s=50, alpha=0.7, color=color)

axs[0, 1].set_xlabel('Cortical Map of tVNS-induced Changes', fontsize=12)
axs[0, 1].set_ylabel('Receptor Density Map', fontsize=12)

# Plot for Sandiego2015
axs[1, 0].set_title('Sandiego2015', fontsize=14)
for roi_index in range(len(mean_img_fslr_parc)):
    color = map_roi_to_color(roi_index)  # Map ROI to corresponding color
    axs[1, 0].scatter(mean_img_fslr_parc[roi_index], des_anno_fslr_parc_6[roi_index],
                label='Cortical', s=50, alpha=0.7, color=color)

axs[1, 0].set_xlabel('Cortical Map of tVNS-induced Changes', fontsize=12)
axs[1, 0].set_ylabel('Receptor Density Map', fontsize=12)

# Plot for Smith2017
axs[1, 1].set_title('Smith2017', fontsize=14)
for roi_index in range(len(mean_img_fslr_parc)):
    color = map_roi_to_color(roi_index)  # Map ROI to corresponding color
    axs[1, 1].scatter(mean_img_fslr_parc[roi_index], des_anno_fslr_parc_7[roi_index],
                label='Cortical', s=50, alpha=0.7, color=color)

axs[1, 1].set_xlabel('Cortical Map of tVNS-induced Changes', fontsize=12)
axs[1, 1].set_ylabel('Receptor Density Map', fontsize=12)

# Optional: Customize spines
for ax in axs.flat:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('grey')
        spine.set_linewidth(0.5)

# Setting common title
fig.suptitle('Correlation of Cortical Map of tVNS-induced Changes with D2/3 Receptor Distribution (Group Level)', fontsize=16)

# Creating legend with subnetwork names and colors
legend_handles = []
for subnetwork_name, color in subnetwork_colors.items():
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor=color, label=subnetwork_name))

# Show legend in the upper right part of the second plot
axs[0, 1].legend(handles=legend_handles, loc='upper right', fontsize=10)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Save the subnetwork plot
fig_path = os.path.join(data_directory, 'subnetworks_cortical_roi_dopamine.png')
plt.savefig(fig_path)
plt.show()

