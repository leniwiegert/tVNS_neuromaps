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

#-------- PREPARE DATA --------#

# Directory containing the volume files
data_directory = '/home/leni/Documents/Master/data/'
#data_directory = '/home/neuromadlab/tVNS_project/data/'

annotation_sources = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

'''
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


# hesse instead of all 11 maps

#-------- RANDOMIZATION --------#

# Number of iterations
num_iterations = 1000

# List to store correlation values
corr_values_rand = []

# Loop for the specified number of iterations
for iteration in range(num_iterations):
    # Assuming non_rand_mask_data_dict is a dictionary with 41 arrays
    for key, volume_data in non_rand_mask_data_dict.items():
        # Generate a random array of 1s and -1s with the same shape as the volume data
        random_multiplier = np.random.choice([-1, 1], size=volume_data.shape)

        # Multiply each value in the volume data by the corresponding value in the random array
        randomized_data = volume_data * random_multiplier

        # Update the original data array with the randomized values
        non_rand_mask_data_dict[key] = randomized_data

        # np.eye(4) creates a 4x4 identity matrix, used as the affine transformation matrix when creating the new NIfTI image
        # Adjust if needed
        affine_matrix = np.eye(4)

        # Create a new NIfTI image with the randomized data
        randomized_img = nib.Nifti1Image(randomized_data, affine_matrix)

        # Save the randomized image to a new file
        randomized_file_path = os.path.join(data_directory, f"randomized_{key}.nii.gz")
        nib.save(randomized_img, randomized_file_path)

        # Print a message indicating the randomization for each key
        #print(f"Randomized data for key: {key}")


    #-------- CALCULATION OF MEANS --------#

    # List all randomized NIfTI files
    rand_files = [f for f in os.listdir(data_directory) if f.startswith('randomized_') and f.endswith('.nii.gz')]

    # Initialize an array to store the data of each randomized image
    rand_data_list = []

    # Load data from each randomized NIfTI image
    for rand_file in rand_files:
        rand_file_path = os.path.join(data_directory, rand_file)
        img = nib.load(rand_file_path)
        rand_data = img.get_fdata()
        rand_data_list.append(rand_data)

        # Compute the mean of the randomized data along the first axis (axis=0)
        mean_rand_data = np.mean(rand_data_list, axis=0)

        # Assuming the first randomized image has the correct affine matrix
        affine_matrix = nib.load(os.path.join(data_directory, rand_files[0])).affine

        # Create a new NIfTI image with the mean data
        mean_rand_img = nib.Nifti1Image(mean_rand_data, affine_matrix)

        # Save the mean NIfTI image to a file
        mean_rand_file_path = os.path.join(data_directory, "mean_randomized.nii.gz")
        nib.save(mean_rand_img, mean_rand_file_path)



    #-------- SPATIAL CORRELATIONS - RANDOMIZED DATA --------#

    # Fetch desired annotation (add description, space, and density if needed for identification)
    anno = fetch_annotation(source='hesse2017')

    # Load the mean image of the randomized data
    mean_rand_img = os.path.join(data_directory, 'mean_randomized.nii.gz')
    mean_rand_img = nib.load(mean_rand_img)

    # Resample the original data to match the annotation space
    data_res_rand, anno_res = resample_images(src=mean_rand_img, trg=anno,
                                                src_space='MNI152', trg_space='MNI152',
                                                method='linear', resampling='downsample_only')

    # Compare resampled original data with neuromaps annotation using the compare_images function
    corr_rand = compare_images(data_res_rand, anno_res, metric='pearsonr')

    # Print the correlation result as needed
    #print(f'Correlation with Randomized Mean Image: {corr_original}')

    # Save the correlation value to the list
    corr_values_rand.append(corr_rand)

# Print the list of correlation values
#print("Correlation values for each iteration:", corr_values_rand)
'''


#-------- SPATIAL CORRELATION - ORIGINAL DATA --------#

# Original correlation value for hesse2017

# Fetch desired annotation (add description, space, and density if needed for identification)
anno = fetch_annotation(source='hesse2017')

# previous code, keep it
# Load the mean image of the randomized data
mean_orig_img = os.path.join(data_directory, 'combined_mask.nii.gz')
mean_orig_img = nib.load(mean_orig_img)
mean_orig_data = mean_orig_img.get_fdata()

# Resample the original data to match the annotation space
data_res_rand, anno_res = resample_images(src=mean_orig_img, trg=anno,
                                            src_space='MNI152', trg_space='MNI152',
                                            method='linear', resampling='downsample_only')

# Compare resampled original data with neuromaps annotation using the compare_images function
#corr_value_orig = compare_images(data_res_rand, anno_res, metric='pearsonr')

# Print the correlation result as needed
#print(f'Correlation with Original Mean Image: {corr_value_orig}')




### TEST for all 11 brain maps ###

'''
# seemingly correct version (following version for testing)

# Manually define colors for each histogram
hist_colors = ['blue', 'blue', 'olive', 'green', 'green', 'green', 'green', 'cyan', 'red', 'orange', 'yellow']

# Create list for saving the r-value, the p-value and the nulls for each correlation
corr_vals_mean_list = []
pval_mean_list = []

# Define the number of columns you want
num_columns = 2

# Calculate the number of rows based on the number of annotation sources and the number of columns
num_rows = -(-len(annotation_sources) // num_columns)  # Ceiling division to ensure all items are shown

# All histograms in one figure
fig, axs = plt.subplots(num_rows, num_columns, figsize=(12, 12), sharex=False, sharey=False)

# Plotting loop
for i, source in enumerate(annotation_sources):
    row = i // num_columns
    col = i % num_columns

    # Fetch annotation
    annotation = fetch_annotation(source=source)

    # Resample the original data to match the annotation space
    data_res, anno_res = resample_images(src=mean_rand_img, trg=annotation,
                                         src_space='MNI152', trg_space='MNI152',
                                         method='linear', resampling='downsample_only')

    # Calculate spatial correlation
    corr_val_mean = compare_images(data_res, anno_res, metric='pearsonr', ignore_zero=True,
                                   nan_policy='omit')

    # Calculate the correlation value for the original data
    data_res_orig, anno_res_orig = resample_images(src=mean_orig_img, trg=annotation,
                                                   src_space='MNI152', trg_space='MNI152',
                                                   method='linear', resampling='downsample_only')
    corr_val_orig = compare_images(data_res_orig, anno_res_orig, metric='pearsonr', ignore_zero=True,
                                   nan_policy='omit')

    # Determine the correct subplot to use
    if num_rows == 1:
        subplot_ax = axs[col]
    else:
        subplot_ax = axs[row, col]

    # Create a histogram for the current map
    sns.histplot(data=corr_values_rand_all_maps, color=hist_colors[i], edgecolor='black', kde=True,
                 ax=subplot_ax, legend=False)

    # Plot vertical line for the original correlation value
    subplot_ax.axvline(corr_val_orig, color='red', linestyle='dashed', linewidth=2, label='Original r-Value')

    subplot_ax.set_xlabel('Spatial Correlation Values')
    subplot_ax.set_ylabel('Frequency')
    subplot_ax.set_title(source)

# Set the limits of the x and y axes manually
plt.subplots_adjust(wspace=0.3, hspace=3)  # adds twice the default space between the plots

# Add custom legend for the colors
unique_colors = list(set(hist_colors))  # Get unique colors
legend_labels = ['NE', 'D1', 'D2/3', 'DAT', '5-HTT', '5-HTb', '5-HT6']  # Corresponding labels for unique colors

# Create legend handles and labels
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label)
                  for color, label in zip(unique_colors, legend_labels)]

# Add custom legend for the similarity value
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.show()



# Original correlation value for all annotations
# List to store correlation values for each brain map
corr_values_orig = []

# Iterate over each brain map
for source in annotation_sources:
    # Fetch annotation
    annotation = fetch_annotation(source=source)

    # Resample the original data to match the annotation space
    data_res_orig, anno_res = resample_images(src=mean_orig_img, trg=annotation,
                                              src_space='MNI152', trg_space='MNI152',
                                              method='linear', resampling='downsample_only')

    # Calculate spatial correlation
    corr_val_orig = compare_images(data_res_orig, anno_res, metric='pearsonr', ignore_zero=True,
                                   nan_policy='omit')

    # Append the correlation value to the list
    corr_values_orig.append(corr_val_orig)

# Print or use the list of correlation values for each brain map
print("Correlation values for each brain map:", corr_values_orig)


########### OPTIMIZED PLOTTING ###########

#--------- LOOP FOR SC FOR MEAN IMAGE WITH 11 MAPS ---------#

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
#annotation_sources = ['alarkurtti2015', 'ding2010', 'fazio2016', 'gallezot2010', 'hesse2017',
#                      'jaworska2020', 'kaller2017', 'radnakrishnan2018', 'sandiego2015', 'sasaki2012', 'smith2017']
annotation_sources = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

# Manually define colors for each histogram
hist_colors = ['blue', 'blue', 'olive', 'green', 'green', 'green', 'green', 'cyan', 'red', 'orange', 'yellow']

# Create list for saving the r-value, the p-value and the nulls for each correlation
corr_vals_mean_list = []
pval_mean_list = []
#corr_nulls_mean_list = []

# Define the number of columns you want
num_columns = 2

# Calculate the number of rows based on the number of annotation sources and the number of columns
num_rows = -(-len(annotation_sources) // num_columns)  # Ceiling division to ensure all items are shown

# All histograms in one figure
fig, axs = plt.subplots(num_rows, num_columns, figsize=(12, 12), sharex=False, sharey=False)
'''

num_iterations = 1000

# Manually define colors for each histogram
hist_colors = ['blue', 'blue', 'olive', 'green', 'green', 'green', 'green', 'cyan', 'red', 'orange', 'yellow']

# Create list for saving the r-value, the p-value and the nulls for each correlation
corr_vals_mean_list = []
pval_mean_list = []
corr_values_rand_all_maps = []
corr_values_orig = []

# Define the number of columns you want
num_columns = 2

# Calculate the number of rows based on the number of annotation sources and the number of columns
num_rows = -(-len(annotation_sources) // num_columns)  # Ceiling division to ensure all items are shown

# All histograms in one figure
fig, axs = plt.subplots(num_rows, num_columns, figsize=(12, 12), sharex=False, sharey=False)

# Plotting loop
for i, source in enumerate(annotation_sources):
    print(f"Processing brain map {i + 1}/{len(annotation_sources)}: {source}")
    row = i // num_columns
    col = i % num_columns

    # Fetch annotation
    annotation = fetch_annotation(source=source)

    # Initialize a list to store correlation values for this brain map
    corr_values_rand_single_map = []
    randomized_data_dict = {}

    # Loop for the specified number of iterations
    print("Starting iteration loop")
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")

        # Randomize data and calculate correlation for this iteration
        random_multiplier = np.random.choice([-1, 1], size=mean_orig_img.shape)
        randomized_data = mean_orig_data * random_multiplier

        # Create a new NIfTI image with the randomized data
        affine_matrix = np.eye(4)  # Assuming identity matrix
        randomized_img = nib.Nifti1Image(randomized_data, affine_matrix)

        # Resample the randomized data to match the annotation space
        data_res_rand, anno_res = resample_images(src=randomized_img, trg=annotation,
                                                  src_space='MNI152', trg_space='MNI152',
                                                  method='linear', resampling='downsample_only')

        # Calculate spatial correlation
        corr_val_rand = compare_images(data_res_rand, anno_res, metric='pearsonr', ignore_zero=True,
                                       nan_policy='omit')

        # Append the correlation value to the list for this brain map
        corr_values_rand_single_map.append(corr_val_rand)

        # Append the list of correlation values for this brain map to the overall list
    corr_values_rand_all_maps.append(corr_values_rand_single_map)
    print("Iteration loop completed")

# original image

    # Calculate the correlation value for the original data
    data_res_orig, anno_res_orig = resample_images(src=mean_orig_img, trg=annotation,
                                                   src_space='MNI152', trg_space='MNI152',
                                                   method='linear', resampling='downsample_only')
    corr_val_orig = compare_images(data_res_orig, anno_res_orig, metric='pearsonr', ignore_zero=True,
                                   nan_policy='omit')
    corr_values_orig.append(corr_val_orig)

'''
# p-value loop
p_values = []

# Iterate over each list of randomized correlation values for each brain map
for corr_values_rand_single_map in corr_values_rand_all_maps:
    p_map_values = []  # List to store p-values for the current brain map

    # Iterate over the original correlation values for the current brain map
    for corr_val_orig in corr_values_orig:
        # Calculate the p-value for each original correlation value
        p_value = percentileofscore(corr_values_rand_single_map, corr_val_orig)
        p_map_values.append(p_value)

    # Append the list of p-values for the current brain map to the overall list
    p_values.append(p_map_values)

# Print or use p_values as needed
print(p_values)


# Plotting loop to include p-values and original correlation lines
for i, source in enumerate(annotation_sources):
    # Fetch annotation
    annotation = fetch_annotation(source=source)

    # Determine the correct subplot to use
    row = i // num_columns
    col = i % num_columns
    if num_rows == 1:
        subplot_ax = axs[col]
    else:
        subplot_ax = axs[row, col]

    # Create a histogram for the current map
    sns.histplot(data=corr_values_rand_all_maps[i], color=hist_colors[i], edgecolor='black', kde=True,
                 ax=subplot_ax, legend=False)

    # Plot vertical line for the original correlation value
    subplot_ax.axvline(corr_val_orig, color='red', linestyle='dashed', linewidth=2, label='Original r-Value')

    # Add p-value as text annotation
    p_value = p_values[i]
    subplot_ax.text(1.02, 0.5, f'p = {p_value[0]:.4f}', transform=subplot_ax.transAxes, va='center', ha='left')

    subplot_ax.set_xlabel('Spatial Correlation Values')
    subplot_ax.set_ylabel('Frequency')
    subplot_ax.set_title(source)

# Set the limits of the x and y axes manually
plt.subplots_adjust(wspace=0.3, hspace=3)  # adds twice the default space between the plots

# Add custom legend for the colors
unique_colors = list(set(hist_colors))  # Get unique colors
legend_labels = ['NE', 'D1', 'D2/3', 'DAT', '5-HTT', '5-HTb', '5-HT6']  # Corresponding labels for unique colors

# Create legend handles and labels
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label)
                  for color, label in zip(unique_colors, legend_labels)]

# Add custom legend for the similarity value
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.show()

print(type(p_values))
print(p_values)
'''
#p_values = []

# Plotting loop to include p-values and original correlation lines
for i, source in enumerate(annotation_sources):
    # Fetch annotation
    annotation = fetch_annotation(source=source)

    # Determine the correct subplot to use
    row = i // num_columns
    col = i % num_columns
    if num_rows == 1:
        subplot_ax = axs[col]
    else:
        subplot_ax = axs[row, col]

    # Calculate the correlation value for the original data
    data_res_orig, anno_res_orig = resample_images(src=mean_orig_img, trg=annotation,
                                                   src_space='MNI152', trg_space='MNI152',
                                                   method='linear', resampling='downsample_only')
    corr_val_orig = compare_images(data_res_orig, anno_res_orig, metric='pearsonr', ignore_zero=True,
                                   nan_policy='omit')

    # Create a histogram for the current map
    sns.histplot(data=corr_values_rand_all_maps[i], color=hist_colors[i], edgecolor='black', kde=True,
                 ax=subplot_ax, legend=False)

    # Plot vertical line for the original correlation value
    subplot_ax.axvline(corr_val_orig, color='red', linestyle='dashed', linewidth=2, label='Original r-Value')

    # Add p-value as text annotation
    #p_value = p_values[i]
    #subplot_ax.text(1.02, 0.5, f'p = {p_value[0]:.4f}', transform=subplot_ax.transAxes, va='center', ha='left')

    subplot_ax.set_xlabel('Spatial Correlation Values')
    subplot_ax.set_ylabel('Frequency')
    subplot_ax.set_title(source)

# Set the limits of the x and y axes manually
plt.subplots_adjust(wspace=0.3, hspace=3)  # adds twice the default space between the plots

# Add custom legend for the colors
unique_colors = list(set(hist_colors))  # Get unique colors
legend_labels = ['NE', 'D1', 'D2/3', 'DAT', '5-HTT', '5-HTb', '5-HT6']  # Corresponding labels for unique colors

# Create legend handles and labels
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label)
                  for color, label in zip(unique_colors, legend_labels)]

# Add custom legend for the similarity value
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.show()

p_values = []
for i in range(len(annotation_sources)):
    # Get the original correlation value for this brain map
    corr_val_orig = corr_values_orig[i]

    # Get the list of randomized correlation values for this brain map
    corr_values_rand_single_map = corr_values_rand_all_maps[i]

    # Calculate p-value for this brain map
    p_value = percentileofscore(corr_values_rand_single_map, corr_val_orig)
    p_values.append(p_value)
    print(p_values)

'''
# this verison seems to have issues
 #OLD VERSION WITHOUT P-VALUES

for i, source in enumerate(annotation_sources):
    row = i // num_columns
    col = i % num_columns

    # Fetch annotation
    annotation = fetch_annotation(source=source)

    # Resample the original data to match the annotation space
    data_res, anno_res = resample_images(src=mean_rand_img, trg=annotation,
                                         src_space='MNI152', trg_space='MNI152',
                                         method='linear', resampling='downsample_only')

    # Calculate spatial correlation
    corr_val_mean = compare_images(data_res, anno_res, metric='pearsonr', ignore_zero=True,
                                           nan_policy='omit')
    
    

    # Determine the correct subplot to use
    if num_rows == 1:
        subplot_ax = axs[col]
    else:
        subplot_ax = axs[row, col]

    # Create a histogram for the current map
    sns.histplot(data=corr_values_rand_all_maps, color=hist_colors[i], edgecolor='blue', kde=True,
                 ax=subplot_ax, legend=False)
    # color=cm(i / len(annotation_sources))

    # Plot vertical line for the similarity value
    subplot_ax.axvline(corr_val_mean, color='red', linestyle='dashed', linewidth=2, label='Original r-Value')

    # Add p-value as text annotation in the top right part of each histogram
    #subplot_ax.text(0.95, 0.95, f'p = {p_values_list[i]:.3f}', transform=subplot_ax.transAxes, ha='right', va='top',
    #                bbox=dict(facecolor='white', alpha=0.5))

    subplot_ax.set_xlabel('Spatial Correlation Values')
    subplot_ax.set_ylabel('Frequency')
    subplot_ax.set_title(source)

# Set the limits of the x and y axes manually
plt.subplots_adjust(wspace=0.3, hspace=3)  # adds twice the default space between the plots

# Add custom legend for the colors
unique_colors = list(set(hist_colors))  # Get unique colors
legend_labels = ['NE', 'D1', 'D2/3', 'DAT', '5-HTT', '5-HTb', '5-HT6']  # Corresponding labels for unique colors

# Create legend handles and labels
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label)
                  for color, label in zip(unique_colors, legend_labels)]

# Add custom legend for the similarity value
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.show()
'''


