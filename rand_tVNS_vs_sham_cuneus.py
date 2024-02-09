# AUTHOR: Lena Wiegert
# Permutation of the tVNS effect

import numpy as np
from neuromaps.datasets import fetch_annotation
from neuromaps.resampling import resample_images
from neuromaps.stats import compare_images
from nilearn import image as nli
import os
import nibabel as nib
import matplotlib.pyplot as plt
# from tqdm import tqdm


#-------- PREPARE DATA --------#

# Directory containing the volume files
data_directory = '/Users/leni/Documents/Master/Data/'

# List of volume files in the directory
volume_files = [f for f in os.listdir(data_directory) if f.startswith('volume_') and f.endswith('.nii')]

# Specify the gray matter mask file
#gray_matter_mask_file = '/Users/leni/Documents/Master/Data/out_GM_p_0_15.nii'
gray_matter_mask_file = os.path.join(data_directory, 'out_GM_p_0_15.nii')
gray_matter_mask = nib.load(gray_matter_mask_file)


#-------- GRAY MATTER MASK --------#

# Create an empty dictionary to store the non_rand mask data for each volume
non_rand_mask_data_dict = {}

# Iterate over each volume file
for volume_file in volume_files:
    # Load the volume image
    volume_path = os.path.join(data_directory, volume_file)
    img = nib.load(volume_path)

    # Resample gray matter mask to match the resolution of the volume image
    gray_matter_mask_resampled = nli.resample_to_img(gray_matter_mask, img)

    # Create a new mask by keeping only non-NaN values in both masks
    non_rand_mask_data = np.where(np.isnan(img.get_fdata()), gray_matter_mask_resampled.get_fdata(), img.get_fdata())

    # Save the non_rand mask data for this volume in the dictionary
    non_rand_mask_data_dict[volume_file] = non_rand_mask_data

    # Create a new image with the non_rand mask
    non_rand_mask_img = nib.Nifti1Image(non_rand_mask_data.astype(np.float32), img.affine)

    # Save the non_rand mask image
    #non_rand_mask_img.to_filename(f'/Users/leni/Documents/Master/Data/{volume_file}_non_rand_mask.nii.gz')
    non_rand_mask_img.to_filename(os.path.join(data_directory, f"{volume_file}_non_rand_mask.nii.gz"))
    print(f"File saved: {volume_file}_non_rand_mask.nii.gz")

    # Save the randomized data array for this volume in the dictionary
    #rand_data_arrays_gm[volume_file] = rand_mask_data

    # Optionally, save the randomized mask data as a NIfTI file
    #rand_mask_img = nib.Nifti1Image(rand_mask_data.astype(np.float32), img.affine)
    #rand_mask_img.to_filename(f'/Users/leni/Documents/Master/Data/{volume_file}_rand_mask.nii.gz')


# List of annotation sources
annotation_sources = ['alarkurtti2015', 'ding2010', 'fazio2016', 'gallezot2010', 'hesse2017',
                      'jaworska2020', 'kaller2017', 'radnakrishnan2018', 'sandiego2015', 'sasaki2012', 'smith2017']

# Number of iterations for randomization
num_iterations = 2  # Adjust as needed

# List to store p-values
p_values = []

# List to store correlation values for the original data
corr_values_orig = []

# List to store correlation values for the randomized data
corr_values_rand = []

# Loop through each annotation source
for source in annotation_sources:
    print(f"\n----- Annotation Source: {source} -----")

    # Fetch desired annotation
    anno = fetch_annotation(source=source)

    # Load the mean image of the original data
    #mean_orig_img = nib.load(f'/Users/leni/Documents/Master/Data/combined_mask.nii.gz')
    mean_orig_img = nib.load(os.path.join(data_directory, 'combined_mask.nii.gz'))

    # Resample the original data to match the annotation space
    data_res_orig, anno_res = resample_images(src=mean_orig_img, trg=anno,
                                              src_space='MNI152', trg_space='MNI152',
                                              method='linear', resampling='downsample_only')

    # Compare resampled original data with neuromaps annotation
    corr_value_orig = compare_images(data_res_orig, anno_res, metric='pearsonr')

    # Store correlation value for the original data
    corr_values_orig.append(corr_value_orig)

    # Print the correlation result for the original data
    print(f'Correlation with Original Mean Image: {corr_value_orig}')

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

        # Resample the original data to match the annotation space
        data_res_rand, anno_res = resample_images(src=mean_rand_img, trg=anno,
                                                  src_space='MNI152', trg_space='MNI152',
                                                  method='linear', resampling='downsample_only')

        # Compare resampled original data with neuromaps annotation
        corr_random = compare_images(data_res_rand, anno_res, metric='pearsonr')

        # Print the correlation result for the randomized data
        print(f'Correlation with Randomized Mean Image (Iteration {iteration + 1}): {corr_random}')

        # Store correlation value for the randomized data
        corr_values_rand.append(corr_random)

    # Calculate p-value
    p_value = (np.sum(np.abs(corr_values_rand) >= np.abs(corr_value_orig)) + 1) / (len(corr_values_rand) + 1)
    print(f'P-value for {source}: {p_value}')
    p_values.append((source, p_value))

# Keep only the first 11 correlation values for the original data (change depending on the nr. of annos)
#corr_values_orig = corr_values_orig[:11]

# Print the list of correlation values for the original data
print("Correlation values for the original data:", corr_values_orig)

# Print the list of correlation values for the randomized data
print("Correlation values for the randomized data:", corr_values_rand)

# Print the p-values with the corresponding annotation sources
print("List of p-values:")
for source, p_value in p_values:
    print(f"{source}: {p_value}")

