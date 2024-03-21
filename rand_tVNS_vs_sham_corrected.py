# AUTHOR: Lena Wiegert
# Permutation of the tVNS effect
import neuromaps
import numpy as np
import seaborn as sns
from neuromaps.datasets import fetch_annotation
from neuromaps.nulls import alexander_bloch
from neuromaps.resampling import resample_images
from neuromaps.stats import compare_images
from nilearn import image as nli
import os
import nibabel as nib
import matplotlib.pyplot as plt
# from tqdm import tqdm


#-------- PREPARE DATA --------#

# Directory containing the volume files
data_directory = '/home/neuromadlab/tVNS_project/data/'

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
    non_rand_mask_img.to_filename(f'/home/neuromadlab/tVNS_project/data/{volume_file}_non_rand_mask.nii.gz')
    print(f"File saved: {volume_file}_non_rand_mask.nii.gz")

    # Save the randomized data array for this volume in the dictionary
    #rand_data_arrays_gm[volume_file] = rand_mask_data

    # Optionally, save the randomized mask data as a NIfTI file
    #rand_mask_img = nib.Nifti1Image(rand_mask_data.astype(np.float32), img.affine)
    #rand_mask_img.to_filename(f'/Users/leni/Documents/Master/Data/{volume_file}_rand_mask.nii.gz')



#-------- RANDOMIZATION --------#

# Number of iterations
num_iterations = 100

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
        print(f"Randomized data for key: {key}")


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
    corr_original = compare_images(data_res_rand, anno_res, metric='pearsonr')

    # Print the correlation result as needed
    print(f'Correlation with Randomized Mean Image: {corr_original}')

    # Save the correlation value to the list
    corr_values_rand.append(corr_original)

# Print the list of correlation values
print("Correlation values for each iteration:", corr_values_rand)



#-------- SPATIAL CORRELATION - ORIGINAL DATA --------#

# Calculate spatial correlation of the mean image of the original data or get it from:
# https://docs.google.com/spreadsheets/d/1B0r_KWS_hOjlmUno4ZjLTa3ownrMCHXPvZ1i6DoqL8o/edit#gid=493335622

# Fetch desired annotation (add description, space, and density if needed for identification)
anno = fetch_annotation(source='hesse2017')

# previous code, keep it
# Load the mean image of the randomized data
mean_orig_img = os.path.join(data_directory, 'combined_mask.nii.gz')
mean_orig_img = nib.load(mean_orig_img)

# Resample the original data to match the annotation space
data_res_rand, anno_res = resample_images(src=mean_orig_img, trg=anno,
                                                                   src_space='MNI152', trg_space='MNI152',
                                                                   method='linear', resampling='downsample_only')

# Compare resampled original data with neuromaps annotation using the compare_images function
corr_value_orig = compare_images(data_res_rand, anno_res, metric='pearsonr')

# Print the correlation result as needed
print(f'Correlation with Original Mean Image: {corr_value_orig}')

'''
#-------- HISTOGRAM --------#

# Plotting the histogram for permutation test distribution
plt.figure(figsize=(12, 6))

# Create a histogram of the original and randomized correlation values
plt.hist([corr_values_rand], bins=30, color=['blue'],
         alpha=0.7, label=['Randomized Data'], edgecolor='black', stacked=True)

# Calculate mean correlation value of the original data
#mean_original = np.mean(corr_value_orig)

# Adding a line for the original correlation value (either with mean_original or from the link)
#plt.axvline(x=mean_original, color='red', linestyle='dashed', linewidth=2, label='Original Data')
#plt.axvline(x=-0.085, color='red', linestyle='dashed', linewidth=2, label='Original Data')
plt.axvline(x=corr_value_orig, color='red', linestyle='dashed', linewidth=2, label='Original Data')

# Adding labels and title for the histogram
plt.xlabel('Correlation Value')
plt.ylabel('Frequency')
plt.title('Permutation Test Distribution of Original and Randomized Values')
plt.legend()
plt.grid(True)

# Display the histogram
plt.tight_layout()
plt.show()

'''


########### OPTIMIZED PLOTTING ###########

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
corr_values_rand_list = []

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

''' #OLD VERSION WITHOUT P-VALUES
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
    corr_val_mean = neuromaps.stats.compare_images(data_res, anno_res, metric='pearsonr', ignore_zero=True,
                                           nan_policy='omit')

    # Append to lists
    corr_vals_mean_list.append(corr_val_mean)
    corr_values_rand_list.append(corr_values_rand)

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
    sns.histplot(data=corr_values_rand_list, color=hist_colors[i], edgecolor='black', kde=True,
                 ax=subplot_ax, legend=False)
    # color=cm(i / len(annotation_sources))

    # Plot vertical line for the similarity value
    #subplot_ax.axvline(corr_val_mean, color='red', linestyle='dashed', linewidth=2, label='Original r-Value')

    # Add p-value as text annotation in the top right part of each histogram
    #subplot_ax.text(0.95, 0.95, f'p = {pval_mean:.3f}', transform=subplot_ax.transAxes, ha='right', va='top',
    #                bbox=dict(facecolor='white', alpha=0.5))

    subplot_ax.set_xlabel('Spatial Correlation Values')
    subplot_ax.set_ylabel('Frequency')
    subplot_ax.set_title(source)

# Remove empty subplots if the number of sources is not a multiple of the number of columns
for i in range(len(annotation_sources), num_rows * num_columns):
    fig.delaxes(axs.flatten()[i])

# Set the limits of the x and y axes manually
#axs[i].axis([0, 1, 0, 1])
plt.subplots_adjust(wspace=0.3, hspace=3) # adds twice the default space between the plots

#plt.tight_layout()
plt.show()
'''


#NEW VERSION WITH P-VALUES
# Loop through your data and calculate p-values
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
    corr_val_mean = neuromaps.stats.compare_images(data_res, anno_res, metric='pearsonr', ignore_zero=True,
                                                   nan_policy='omit')

    # Generate your null model data (assuming null_model_data is defined)
    #null_model_data = generate_null_model(data_res)  # Replace generate_null_model with your null model generation function
    null_model_data = alexander_bloch(data_res, atlas='MNI152', density='3mm', parcellation=None)

    # Calculate correlation between original data and null model
    null_corr = np.corrcoef(data_res.ravel(), null_model_data.ravel())[0, 1]

    # Calculate p-value
    p_value = (np.sum(np.abs(corr_values_rand_list[i]) >= np.abs(corr_val_mean)) + 1) / (len(corr_values_rand_list[i]) + 1)

    # Append to lists
    corr_vals_mean_list.append(corr_val_mean)
    pval_mean_list.append(p_value)

    # Determine the correct subplot to use
    if num_rows == 1:
        subplot_ax = axs[col]
    else:
        subplot_ax = axs[row, col]

    # Create a histogram for the current map
    sns.histplot(data=corr_values_rand_list[i], color=hist_colors[i], edgecolor='black', kde=True,
                 ax=subplot_ax, legend=False)

    # Plot vertical line for the similarity value
    # subplot_ax.axvline(corr_val_mean, color='red', linestyle='dashed', linewidth=2, label='Original r-Value')

    # Add p-value as text annotation in the top right part of each histogram
    subplot_ax.text(0.95, 0.95, f'p = {p_value:.3f}', transform=subplot_ax.transAxes, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.5))

    subplot_ax.set_xlabel('Spatial Correlation Values')
    subplot_ax.set_ylabel('Frequency')
    subplot_ax.set_title(source)

# Remove empty subplots if the number of sources is not a multiple of the number of columns
for i in range(len(annotation_sources), num_rows * num_columns):
    fig.delaxes(axs.flatten()[i])

# Set the limits of the x and y axes manually
# axs[i].axis([0, 1, 0, 1])
plt.subplots_adjust(wspace=0.3, hspace=3)  # adds twice the default space between the plots

plt.show()

##############################################################################

# old error?
#Traceback (most recent call last):
#  File "/mnt/ghrelin/tvns_neuromaps/rand_tVNS_vs_sham_corrected.py", line 268, in <module>
#    corr_val_mean, pval_mean = neuromaps.stats.compare_images(data_res, anno_res, metric='pearsonr', ignore_zero=True,
#TypeError: cannot unpack non-iterable numpy.float64 object







