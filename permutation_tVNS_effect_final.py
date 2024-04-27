# AUTHOR: Lena Wiegert
# Permutation of the tVNS effect
import neuromaps
import numpy as np
import seaborn as sns
from neuromaps.datasets import fetch_annotation, fetch_atlas
from neuromaps.nulls import alexander_bloch
from neuromaps.resampling import resample_images
from neuromaps.stats import compare_images
from nilearn import image as nli
import os
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.image import resample_img
from scipy.stats import percentileofscore, pearsonr, ttest_1samp

# -------- PREPARE DATA --------#

# Directory containing the volume files
data_directory = '/home/leni/Documents/Master/data/'
# data_directory = '/home/neuromadlab/tVNS_project/data/'

# List of brain maps to calculate correlations with
annotation_sources = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']


# List of volume files in the directory
volume_files = [f for f in os.listdir(data_directory) if f.startswith('volume_') and f.endswith('.nii')]

# Specify the gray matter mask file
gray_matter_mask_file = os.path.join(data_directory, 'out_GM_p_0_15.nii')
gray_matter_mask = nib.load(gray_matter_mask_file)

#-------- GRAY MATTER MASK --------#

# Create an empty dictionary to store the non_rand mask data for each volume
non_rand_mask_data_dict = {}

# Create an empty dictionary to store randomized data arrays for each volume
#rand_data_arrays_gm = {}

# Iterate over each volume file
for volume_file in volume_files:
    # Load the volume image
    volume_path = os.path.join(data_directory, volume_file)
    img = nib.load(volume_path)

    # Resample gray matter mask to match the resolution of the volume image
    gray_matter_mask_resampled = nli.resample_to_img(gray_matter_mask, img)

    # Ensure both masks have the same shape
    #if not np.all(gray_matter_mask_resampled.shape == img.shape):
    #    raise ValueError('Shape of input volume is incompatible.')

    # Create a new mask by keeping only non-NaN values in both masks
    non_rand_mask_data = np.where(np.isnan(img.get_fdata()), gray_matter_mask_resampled.get_fdata(), img.get_fdata())

    # Save the non_rand mask data for this volume in the dictionary
    non_rand_mask_data_dict[volume_file] = non_rand_mask_data

    # Create a new image with the non_rand mask
    non_rand_mask_img = nib.Nifti1Image(non_rand_mask_data.astype(np.float32), img.affine)

    # Save the non_rand mask image
    #non_rand_mask_img.to_filename(f'/home/leni/Documents/Master/data/{volume_file}_non_rand_mask.nii.gz')
    non_rand_mask_img.to_filename(f'{data_directory}{volume_file}_non_rand_mask.nii.gz')
    #print(f"File saved: {volume_file}_non_rand_mask.nii.gz")

    # Save the randomized data array for this volume in the dictionary
    #rand_data_arrays_gm[volume_file] = rand_mask_data

    # Optionally, save the randomized mask data as a NIfTI file
    #rand_mask_img = nib.Nifti1Image(rand_mask_data.astype(np.float32), img.affine)
    #rand_mask_img.to_filename(f'/Users/leni/Documents/Master/Data/{volume_file}_rand_mask.nii.gz')



# -------- SPATIAL CORRELATION - ORIGINAL MEAN IMAGE --------#

# Load the original mean image
mean_orig_img = os.path.join(data_directory, 'combined_mask.nii.gz')
mean_orig_img = nib.load(mean_orig_img)
mean_orig_data = mean_orig_img.get_fdata()
affine_matrix = mean_orig_img.affine

# Initialize a dictionary to store correlation values for each brain map
correlation_results_orig = {}

# Iterate over each brain map
for brain_map in annotation_sources:
    # Fetch the desired annotation
    anno = fetch_annotation(source=brain_map)

    # Resample the original mean data to match the annotation space
    data_res_orig, anno_res = resample_images(src=mean_orig_img, trg=anno,
                                              src_space='MNI152', trg_space='MNI152',
                                              method='linear', resampling='downsample_only')

    # Compare resampled original mean data with the brain map annotation using the compare_images function
    corr_orig = compare_images(data_res_orig, anno_res, metric='pearsonr')

    # Store the correlation value
    correlation_results_orig[brain_map] = corr_orig
print(correlation_results_orig)


# -------- SPATIAL CORRELATION - RANDOMIZED MEAN IMAGE --------#

# Spatial correlations of the randomized mean image with 11 maps with permutation

# Initialize a dictionary to store correlation values for each brain map and each permutation
permuted_correlation_results = {brain_map: [] for brain_map in annotation_sources}

# Iterate 1000 times for permutation
for _ in range(1000): # Desired number of permutations
    print(f"Iteration {_ + 1} of 1000")
    # Initialize a list to store correlation values for this permutation
    permutation_corr_values = []

    # Generate a new random multiplier for each volume
    random_multipliers = {key: np.random.choice([-1, 1]) for key in non_rand_mask_data_dict}

    # Randomize the sign of the volume data for each volume
    randomized_data_list = []
    for key, volume_data in non_rand_mask_data_dict.items():
        randomized_data = volume_data * random_multipliers[key]
        randomized_data_list.append(randomized_data)

    # Compute the mean of the randomized data along the first axis (axis=0)
    mean_rand_data = np.mean(randomized_data_list, axis=0)

    # Create a new NIfTI image with the mean data
    mean_rand_img = nib.Nifti1Image(mean_rand_data, affine_matrix)

    # Compute the spatial correlation between the mean randomized data and each brain map annotation
    for brain_map in annotation_sources:
        # Fetch the desired annotation
        anno = fetch_annotation(source=brain_map)

        # Resample the mean randomized data to match the annotation space
        data_res_rand, anno_res = resample_images(src=mean_rand_img, trg=anno,
                                                  src_space='MNI152', trg_space='MNI152',
                                                  method='linear', resampling='downsample_only')

        # Compute the correlation between the resampled mean randomized data and the brain map annotation
        corr_rand = compare_images(data_res_rand, anno_res, metric='pearsonr')

        # Store the correlation value
        permutation_corr_values.append(corr_rand)

    # Append the correlation values for this permutation to the dictionary
    for brain_map, corr_value in zip(annotation_sources, permutation_corr_values):
        permuted_correlation_results[brain_map].append(corr_value)

#------------- SIGNIFICANCE TESTING ------------#

# Initialize a dictionary to store p-values for each brain map
p_values = {}

# Iterate over each brain map
for brain_map in annotation_sources:
    # Retrieve the original correlation value
    original_corr_value = correlation_results_orig[brain_map]

    # Count the number of permuted correlation values that are as or more extreme than the original value
    count_extreme = sum(corr_value >= original_corr_value for corr_value in permuted_correlation_results[brain_map])

    # Calculate the p-value
    p_value = count_extreme / len(permuted_correlation_results[brain_map])

    # Store the p-value
    p_values[brain_map] = p_value

print("P-values:")
print(p_values)


#------------- PLOTTING ------------#

# Manually define colors for each histogram
hist_colors = ['blue', 'blue', 'olive', 'green', 'green', 'green', 'green', 'cyan', 'red', 'orange', 'yellow']

# Define the number of columns you want
num_columns = 2

# Calculate the number of rows based on the number of annotation sources and the number of columns
num_rows = -(-len(annotation_sources) // num_columns)  # Ceiling division to ensure all items are shown

# All histograms in one figure
fig, axs = plt.subplots(num_rows, num_columns, figsize=(12, 12), sharex=False, sharey=False)

for i, source in enumerate(annotation_sources):
    # Determine the correct subplot to use
    row = i // num_columns
    col = i % num_columns
    if num_rows == 1:
        subplot_ax = axs[col]
    else:
        subplot_ax = axs[row, col]

    # Create a histogram for the current map using permuted correlation results
    sns.histplot(data=permuted_correlation_results[source], color=hist_colors[i], edgecolor='black', kde=True,
                 ax=subplot_ax, legend=False)

    subplot_ax.set_xlabel('Spatial Correlation Values')
    subplot_ax.set_ylabel('Frequency')
    subplot_ax.set_title(source)

    # Retrieve the original correlation value for this brain map
    original_corr_value = correlation_results_orig[source]
    # Plot vertical line for the original correlation value
    subplot_ax.axvline(original_corr_value, color='red', linestyle='dashed', linewidth=2, label='Original r-Value')

    p_value = p_values[source]
    subplot_ax.text(0.95, 0.95, f'p = {p_value:.3f}', transform=subplot_ax.transAxes, color='black', fontsize=10,
                    verticalalignment='top', horizontalalignment='right')

# Set the limits of the x and y axes manually
plt.subplots_adjust(wspace=0.3, hspace=3)  # adds twice the default space between the plots

# Add custom legend for the colors
unique_colors = list(set(hist_colors))  # Get unique colors
legend_labels = ['NE', 'D1', 'D2/3', 'DAT', '5-HTT', '5-HTb', '5-HT6']  # Corresponding labels for unique colors

# Create legend handles and labels
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label)
                  for color, label in zip(unique_colors, legend_labels)]

# Add custom legend for the similarity value
fig.legend(handles=legend_handles, labels=legend_labels, loc='upper right')

plt.show()


