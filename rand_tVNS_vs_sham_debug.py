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
gray_matter_mask_file = '/Users/leni/Documents/Master/Data/out_GM_p_0_15.nii'
gray_matter_mask = nib.load(gray_matter_mask_file)


#-------- GRAY MATTER MASK + RANDOMIZATION + CALCULATION OF MEANS--------#

# Create an empty dictionary to store the non_rand mask data for each volume
non_rand_mask_data_dict = {}

# Create an empty dictionary to store randomized data arrays for each volume
rand_data_arrays = {}

# Specify the number of randomizations (10 for testing, 1000 when it works)
num_randomizations = 100

# Iterate over each volume file
for volume_file in volume_files:
    # Load the volume image
    volume_path = os.path.join(data_directory, volume_file)
    img = nib.load(volume_path)

    # Resample gray matter mask to match the resolution of the volume image
    gray_matter_mask_resampled = nli.resample_to_img(gray_matter_mask, img)

    # Ensure both masks have the same shape
    if not np.all(gray_matter_mask_resampled.shape == img.shape):
        raise ValueError('Shape of input volume is incompatible.')

    # Create a new mask by keeping only non-NaN values in both masks
    non_rand_mask_data = np.where(np.isnan(img.get_fdata()), gray_matter_mask_resampled.get_fdata(), img.get_fdata())

    # Save the non_rand mask data for this volume in the dictionary
    non_rand_mask_data_dict[volume_file] = non_rand_mask_data

    # Create a new image with the non_rand mask
    non_rand_mask_img = nib.Nifti1Image(non_rand_mask_data.astype(np.float32), img.affine)

    # Save the non_rand mask image
    non_rand_mask_img.to_filename(f'/Users/leni/Documents/Master/Data/{volume_file}_non_rand_mask.nii.gz')
    print(f"File saved: {volume_file}_non_rand_mask.nii.gz")

    # Save the randomized data array for this volume in the dictionary
    #rand_data_arrays[volume_file] = rand_mask_data

    # Optionally, save the randomized mask data as a NIfTI file
    #rand_mask_img = nib.Nifti1Image(rand_mask_data.astype(np.float32), img.affine)
    #rand_mask_img.to_filename(f'/Users/leni/Documents/Master/Data/{volume_file}_rand_mask.nii.gz')

    # Iterate for the specified number of randomizations
    for randomization_index in range(num_randomizations):
        # Create a randomized copy of the non_rand mask data
        rand_mask_data = non_rand_mask_data * np.random.choice([-1, 1], size=non_rand_mask_data.shape)

        # Save the randomized data array for this volume in the dictionary
        rand_data_arrays[f"{volume_file}_random_{randomization_index}"] = rand_mask_data

        # Optionally, save the randomized mask data as a NIfTI file
        rand_mask_img = nib.Nifti1Image(rand_mask_data.astype(np.float32), img.affine)
        rand_mask_img.to_filename(
            f'/Users/leni/Documents/Master/Data/{volume_file}_random_{randomization_index}_mask.nii.gz')

        # You can add a print statement here if you want to indicate each randomization
        print(f"Randomization {randomization_index + 1} saved for {volume_file}")


# Compute the mean for each volume for original and randomized data
mean_non_rand_data = {volume_file: np.mean(data_array, axis=0) for volume_file, data_array in
                          non_rand_mask_data_dict.items()}
mean_rand_data = {volume_file: np.mean(data_array, axis=0) for volume_file, data_array in
                            rand_data_arrays.items()}
print(mean_non_rand_data)
print(mean_rand_data)



#-------- SPATIAL CORRELATIONS --------#

# Fetch desired annotation (add description, space, and density if needed for identification)
anno = fetch_annotation(source='hesse2017')

# Lists to store correlation values
correlations_original = []
correlations_rand = []

# Non rand mean  sc calculations:
# Iterate over each volume file
for volume_file in volume_files[0:]:
    # Load the resampled original data image
    mean_non_rand_img = nib.load(f'/Users/leni/Documents/Master/Data/{volume_file}_non_rand_mask.nii.gz')

    # Resample the original data to match the annotation space
    data_resampled_original, anno_resampled_original = resample_images(src=mean_non_rand_img, trg=anno,
                                                                       src_space='MNI152', trg_space='MNI152',
                                                                       method='linear', resampling='downsample_only')

    # Compare resampled original data with neuromaps annotation using the compare_images function
    corr_original = compare_images(data_resampled_original, anno_resampled_original, metric='pearsonr')

    # Append the same correlation value for the original data to the list
    correlations_original.extend([corr_original])

    # Optionally, print or use the correlation result as needed
    print(f"Correlation with Original Data for {volume_file}: {corr_original:.3f}")

    # Iterate over each randomized object in mean_rand_data for the current volume
    for randomization_index in range(num_randomizations):
        # Access the specific randomized object for the current volume
        rand_mask_data = rand_data_arrays[f"{volume_file}_random_{randomization_index}"]

        # Convert the randomized data to a NIfTI image (this line can be removed)
        rand_mask_img = nib.Nifti1Image(rand_mask_data.astype(np.float32), img.affine)

        # Resample the randomized data to match the annotation space
        data_resampled_rand, anno_resampled_rand = resample_images(src=rand_mask_img, trg=anno,
                                                                   src_space='MNI152', trg_space='MNI152',
                                                                   method='linear', resampling='downsample_only')

        # Compare resampled randomized data with neuromaps annotation using the compare_images function
        corr_rand = compare_images(data_resampled_rand, anno_resampled_rand, metric='pearsonr')

        # Append the correlation value for the current randomized object to the list
        correlations_rand.append(corr_rand)

        # Optionally, print or use the correlation result as needed
        print(
            f"Correlation with Randomized Data for {volume_file}, Object {randomization_index + 1}: {corr_rand:.3f}")

print(len(correlations_original))
print(len(correlations_rand))



#-------- SCATTER PLOT --------#

# Define volume_labels
volume_labels_original = [f"Volume {i + 1}" for i in range(len(correlations_original))]

# Same volume labels for the randomized data
volume_labels_rand = np.repeat(volume_labels_original, num_randomizations)

# Flatten the list of lists (correlations_randomized)
#correlations_rand_flat = [value for sublist in correlations_rand for value in sublist]

# Plotting the correlation values
plt.figure(figsize=(12, 6))

# Plot original data as dots
plt.scatter(volume_labels_original, correlations_original, label='Original Data', color='blue', alpha=0.7)

# Plot randomized data as dots
plt.scatter(volume_labels_rand, correlations_rand, label='Randomized Data', color='orange', alpha=0.7)

# Adding labels and title for the scatter plot
plt.xlabel('Volume')
plt.ylabel('Correlation Value')
plt.title('Spatial Correlations of Original vs. Randomized Data - Randomization of tVNS Effect')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(True)

# Display the scatter plot
plt.tight_layout()
plt.show()



#-------- HISTOGRAM --------#

# Plotting the histogram for permutation test distribution
plt.figure(figsize=(12, 6))

# Create a histogram of the original and randomized correlation values
plt.hist([correlations_rand], bins=30, color=['blue'],
         alpha=0.7, label=['Randomized Data'], edgecolor='black', stacked=True)

# Adding a line for the original correlation value
plt.axvline(x=correlations_original[0], color='red', linestyle='dashed', linewidth=2, label='Original Data')

# Adding labels and title for the histogram
plt.xlabel('Correlation Value')
plt.ylabel('Frequency')
plt.title('Permutation Test Distribution of Original and Randomized Values')
plt.legend()
plt.grid(True)

# Display the histogram
plt.tight_layout()
plt.show()


