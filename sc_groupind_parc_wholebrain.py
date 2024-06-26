'''
@author: Lena Wiegert

This code compares the spatial correlations of parcellated cortical rs_FC maps of tVNS-induced changes
to the same data as parcellated whole-brain maps. These steps are performed on group and individual level.
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



###################### CORTICAL DATA ######################

######## CORTICAL DATA - GROUP LEVEL ########

#-------- TRANSFORMATION + PARCELLATION --------#

# Import the parcellation maps (Schaefer) in fsLR space
# fsLR32k
parcels_fslr_32k = fetch_schaefer2018('fslr32k')['400Parcels7Networks']
parcels_fslr_32k = dlabel_to_gifti(parcels_fslr_32k)
parcels_fslr_32k = relabel_gifti(parcels_fslr_32k)
# Create parcellaters for fsLR
parc_fsLR = Parcellater(parcels_fslr_32k, 'fslr', resampling_target=None)
# Transform my image to fsLR
mean_img_fslr = transforms.mni152_to_fslr(mean_img_gm, '32k', method='nearest')
#print(mean_img_fslr)
# The output is a tuple of Gifti Images

# Parcellate MY image
mean_img_fslr_parc = parc_fsLR.fit_transform(mean_img_fslr, 'fsLR')
# The output is an array

#-------- SPATIAL NULLS OF THE MEAN IMAGE --------#

# Generate nulls
nulls_mean = alexander_bloch(mean_img_fslr_parc, atlas='fsLR', density='32k', parcellation=parcels_fslr_32k)
#print(nulls_mean)
#print(len(nulls_mean))
# Should be 400

#--------- LOOP FOR SC FOR MEAN IMAGE WITH 11 MAPS ---------#

# List of annotation sources
annotation_sources = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

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

# Convert list to numpy array
corr_values_mean_array = np.array(corr_values_mean)
file_path = os.path.join(data_directory, 'corr_values_mean_cortical.npy')
np.save(file_path, corr_values_mean_array)
print(corr_values_mean_array.shape)

######## CORTICAL DATA - INDIVIDUAL LEVEL ########

# Initialize correlation values list
corr_values_cortical_single = []
# Initialize correlation values dictionary for each brain map
corr_values_cortical_single_maps = {}

# List of brain maps
brain_maps = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
              'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

# Loop through each brain map
for brain_map in brain_maps:
    print(f"Processing brain map: {brain_map}")

    # Initialize correlation values list for the current brain map
    corr_values_cortical_single = []

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
        # Save the masked data
        # nib.save(vol_1_gm, os.path.join(data_directory, 'vol_1_gm.nii'))

        # Transform the single volumes to fsLR
        vol_fslr = transforms.mni152_to_fslr(vol_gm, '32k')
        # Parcellate the single volumes with Schaefer
        vol_fslr_parc = parc_fsLR.fit_transform(vol_fslr, 'fsLR')

        # Transform annotation to fsLR
        anno_fslr = transforms.mni152_to_fslr(anno, '32k')
        # Parcellate annotation with Schaefer
        anno_fslr_parc = parc_fsLR.fit_transform(anno_fslr, 'fsLR')

        # Calculate spatial correlations
        corr = stats.compare_images(vol_fslr_parc, anno_fslr_parc)
        corr_values_cortical_single.append(corr)

    # Save the correlation values for the current brain map
    corr_values_cortical_single_maps[brain_map] = corr_values_cortical_single

    print(f'Here are the correlation values for cortical single subject data with {brain_map}: {corr_values_cortical_single}')
    #print(len(corr_values_cortical_single))

# Save the correlation values for the subcortical single maps
print(corr_values_cortical_single_maps)
file_path_cort = os.path.join(data_directory, 'corr_values_cortical_single_maps.npy')
np.save(file_path_cort, corr_values_cortical_single_maps)




######################  WHOLE BRAIN DATA ######################

######## WHOLE-BRAIN DATA - GROUP LEVEL ########

#-------- LOOP FOR SPATIAL CORRELATIONS OF THE MEAN IMAGE WITH THE WHOLE-BRAIN DATA AND 11 MAPS --------#

annotation_sources = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

corr_values_mean_subcortical = []
p_values_mean_subcortical = []

# Load the mean volumetric image
mean_orig_img = nib.load(f'{data_directory}combined_mask.nii.gz')

#-------- WHOLE-BRAIN MEAN IMAGE PARCELLATION --------#

# Load Nifti brain atlas file
#atlas_directory = '/home/neuromadlab/Tian2020MSA_v1.4/Tian2020MSA/3T/Cortex-Subcortex/MNIvolumetric/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S4_3T_MNI152NLin2009cAsym_2mm.nii.gz'
atlas_path = '/home/leni/Tian2020MSA_v1.4/Tian2020MSA/3T/Cortex-Subcortex/MNIvolumetric/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S4_3T_MNI152NLin2009cAsym_2mm.nii.gz'
atlas_img = nib.load(atlas_path)

# Initialize Parcellater with the Nifti brain atlas
parcellater = Parcellater(parcellation=atlas_img,  space='MNI152')

# Fit the Parcellater
parcellater.fit()

# Parcellate the resampled MNI152 data using the atlas
parcellated_subcort_mean_data = parcellater.transform(mean_orig_img, space='MNI152')
print(parcellated_subcort_mean_data)

# Check the shape of the parcellated data
print(f'Shape of the parcellated subcortical data: {parcellated_subcort_mean_data.shape}')

# r-values mean cortical
corr_values_mean_cortical = corr_values_mean    # Shape (11,)
print(corr_values_mean_cortical)

#### Correlation with receptor maps:

for source in annotation_sources:
    # Fetch annotation
    annotation = fetch_annotation(source=source)
    # Parcellate annotation with Tian
    annotation_mni152_parc = parcellater.transform(annotation, space='MNI152')

    # Calculate spatial correlation and p-value of subcortical data
    corr_mean = stats.compare_images(parcellated_subcort_mean_data, annotation_mni152_parc)

    # Append to list
    corr_values_mean_subcortical.append(corr_mean)
    #print(corr_values_mean)

# r-values mean subcortical
print(corr_values_mean_subcortical)
corr_values_mean_subcortical_array = np.array(corr_values_mean_subcortical)
print(f'Shape of parcellated subcortical mean data array: {corr_values_mean_subcortical_array.shape}')
file_path = os.path.join(data_directory, 'corr_values_mean_subcortical.npy')
np.save(file_path, corr_values_mean_subcortical_array)

######## WHOLE-BRAIN DATA - INDIVIDUAL LEVEL ########

# Initialize correlation values dictionary for each brain map
corr_values_subcortical_single_maps = {}

# Loop through each brain map
for brain_map in brain_maps:
    print(f"Processing brain map: {brain_map}")

    # Initialize correlation values list for the current brain map
    corr_values_subcortical_single = []

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
        print(f'Shape of the parcellated whole-brain data: {parcellated_subcort_single_data.shape}')

        # Parcellate annotation data with Tian
        anno_mni152_parc = parcellater.transform(anno_img, space='MNI152')

        # Calculate spatial correlations
        corr = stats.compare_images(parcellated_subcort_single_data, anno_mni152_parc)
        corr_values_subcortical_single.append(corr)
        #print(f"Correlation value for volume {i}: {corr}")

    # Save the correlation values for the current brain map
    corr_values_subcortical_single_maps[brain_map] = corr_values_subcortical_single

    print(f'Here are the correlation values for whole-brain single subject data with {brain_map}: {corr_values_subcortical_single}')
    #print(len(corr_values_subcortical_single))

# Save the correlation values for the subcortical single maps
print(corr_values_subcortical_single_maps)
file_path_subcort = os.path.join(data_directory, 'corr_values_subcortical_single_maps.npy')
np.save(file_path_subcort, corr_values_subcortical_single_maps)



'''
# Commented out for individual plotting
######################  PLOTTING ######################

# Cortical mean r-values (11 maps)
print(corr_values_mean_cortical)

# Subcortical mean r-values (11 maps)
print(corr_values_mean_subcortical)

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

# Convert lists to numpy arrays
np_array_subcortical_mean = np.array(corr_values_mean_subcortical)
print(np_array_subcortical_mean)
np_array_cortical_mean = np.array(corr_values_mean_cortical)
print(np_array_cortical_mean)

# Correlation calculation
#correlation_mean = pearsonr(corr_values_mean_cortical, corr_values_mean_subcortical)[0]
#print(f'Here is the correlation value of the mean/group data:{correlation_mean}')


######## PLOTTING: GROUP LEVEL ########

x_values = ['Cortical', 'Whole-brain']

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
plt.show()

'''


######## PLOTTING - INDIVIDUAL LEVEL ########

'''
# Error somewhere, cortical values are the same for each plot

x_values = ['Cortical', 'Whole-brain']

# Define color categories for participants
participant_colors = plt.cm.inferno(np.linspace(0, 1, 41))

for i, (key, value) in enumerate(corr_values_cortical_single_maps.items()):
    plt.figure(figsize=(8, 6))  # Create a new figure for each brain map

    subcortical_values = corr_values_subcortical_single_maps.get(key, None)

    # Check if subcortical_values is None or has different length
    if subcortical_values is None or len(subcortical_values) != 41:
        continue

    for participant_id, color in zip(range(1, 42), participant_colors):
        cortical_value = value[participant_id - 1]
        subcortical_value = subcortical_values[participant_id - 1]

        y_values = np.array([[cortical_value], [subcortical_value]])
        plt.plot(x_values, y_values, marker='o', color=color, label=f'{participant_id}')

    # Customize subplot
    plt.title(key)
    plt.xlabel('Category')
    plt.ylabel('Correlation Value')
    plt.grid(True)
    plt.legend(title='Participant ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Show plot
    plt.show()

'''



'''
# works perfectly :)) 
x_values = ['Cortical', 'Whole-brain']

# Define color categories for participants
participant_colors = plt.cm.inferno(np.linspace(0, 1, 41))

for i, brain_map in enumerate(brain_maps):
    plt.figure(figsize=(8, 6))  # Create a new figure for each brain map

    cortical_values = corr_values_cortical_single_maps.get(brain_map, None)
    subcortical_values = corr_values_subcortical_single_maps.get(brain_map, None)

    # Check if cortical_values or subcortical_values are None or have different lengths
    if cortical_values is None or subcortical_values is None or len(cortical_values) != 41 or len(subcortical_values) != 41:
        continue

    for participant_id, color in zip(range(1, 42), participant_colors):
        cortical_value = cortical_values[participant_id - 1]
        subcortical_value = subcortical_values[participant_id - 1]

        y_values = np.array([[cortical_value], [subcortical_value]])
        plt.plot(x_values, y_values, marker='o', color=color, label=f'{participant_id}')

    # Customize subplot
    plt.title(brain_map)
    plt.xlabel('Category')
    plt.ylabel('Correlation Value')
    plt.grid(True)
    plt.legend(title='Participant ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Show plot
    plt.show()

'''

# Plotting try with mean values on top

x_values = ['Cortical', 'Whole-brain']

# Define color categories for participants
participant_colors = plt.cm.inferno(np.linspace(0, 1, 41))

for i, brain_map in enumerate(brain_maps):
    plt.figure(figsize=(8, 6))  # Create a new figure for each brain map

    cortical_values = corr_values_cortical_single_maps.get(brain_map, None)
    subcortical_values = corr_values_subcortical_single_maps.get(brain_map, None)

    # Check if cortical_values or subcortical_values are None or have different lengths
    if cortical_values is None or subcortical_values is None or len(cortical_values) != 41 or len(subcortical_values) != 41:
        continue

    # Plot mean correlation values for cortical and subcortical regions
    plt.plot(x_values, [corr_values_mean_cortical[i], corr_values_mean_subcortical[i]], marker='o', linestyle='-', color='blue', label='Mean')

    for participant_id, color in zip(range(1, 42), participant_colors):
        cortical_value = cortical_values[participant_id - 1]
        subcortical_value = subcortical_values[participant_id - 1]

        y_values = np.array([[cortical_value], [subcortical_value]])
        plt.plot(x_values, y_values, marker='o', color=color, label=f'{participant_id}')

    # Customize subplot
    plt.title(brain_map)
    plt.xlabel('Category')
    plt.ylabel('Correlation Value')
    plt.grid(True)
    plt.legend(title='Participant ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Show plot
    plt.show()


