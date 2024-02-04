
import numpy as np
from neuromaps.datasets import fetch_annotation
from neuromaps.resampling import resample_images
from neuromaps.stats import compare_images
from nilearn import image as nli
import os
from tqdm import tqdm
import nibabel as nib
import matplotlib.pyplot as plt


# Directory containing the volume files
data_directory = '/Users/leni/Documents/Master/Data/'

# List of volume files in the directory
volume_files = [f for f in os.listdir(data_directory) if f.startswith('volume_') and f.endswith('.nii')]

# Specify the gray matter mask file
gray_matter_mask_file = '/Users/leni/Documents/Master/Data/out_GM_p_0_15.nii'
gray_matter_mask = nib.load(gray_matter_mask_file)

# Specify the number of randomizations
num_randomizations = 5

# Create an empty dictionary to store randomized data arrays for each volume
randomized_data_arrays = {}

# Create an empty dictionary to store non-randomized data arrays for each volume
non_randomized_data_arrays = {}

# Fetch desired annotation (add description, space, and density if needed for identification)
anno = fetch_annotation(source='hesse2017')

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
    combined_mask_data = np.where(np.isnan(img.get_fdata()), gray_matter_mask_resampled.get_fdata(), img.get_fdata())

    # Create a new image with the combined mask
    combined_mask_img = nib.Nifti1Image(combined_mask_data.astype(np.float32), img.affine)

    # ----- Randomized Data -----

    # Create a 3D array to store the results
    randomized_mask_data_array = np.zeros((num_randomizations,) + combined_mask_data.shape)

    # Create a tqdm progress bar for the loop
    for i in tqdm(range(num_randomizations), desc=f"Randomizing {volume_file}", unit="iterations", leave=False):
        # Randomly multiply each value by 1 or -1
        random_signs = np.random.choice([-1, 1], size=combined_mask_data.shape)

        # Apply the random signs to the combined mask data
        randomized_mask_data = combined_mask_data * random_signs

        # Save the result in the array
        randomized_mask_data_array[i, :, :, :] = randomized_mask_data

        # Print a message indicating that randomization for the current volume is finished
        print(f"Randomization for {volume_file} is complete.")

    # Save the randomized data array for this volume in the dictionary
    randomized_data_arrays[volume_file] = randomized_mask_data_array

    # Check if shapes of randomized data arrays are the same
    if not np.all(randomized_data_arrays[volume_file].shape == randomized_data_arrays[volume_files[0]].shape):
        raise ValueError('Shapes of randomized data arrays are not the same.')

    '''# ----- Non-Randomized Data -----

    # Create a 3D array to store the results for non-randomized data
    non_randomized_mask_data_array = np.zeros_like(combined_mask_data)

    # Repeat the non-randomized computation (no random signs) num_randomizations times
    for i in tqdm(range(num_randomizations), desc=f"Non-Randomizing {volume_file}", unit="iterations", leave=False):
        non_randomized_mask_data_array += combined_mask_data

    # Compute the mean for non-randomized data
    non_randomized_mask_data_array /= num_randomizations

    # Save the non-randomized data array for this volume in the dictionary
    non_randomized_data_arrays[volume_file] = non_randomized_mask_data_array

    # Check if shapes of non-randomized data arrays are the same
    if not np.all(non_randomized_data_arrays[volume_file].shape == non_randomized_data_arrays[volume_files[0]].shape):
        raise ValueError('Shapes of non-randomized data arrays are not the same.')

    # Print a message indicating that non-randomization for the current volume is finished
    print(f"Non-Randomization for {volume_file} is complete.")'''

# Compute the mean for each volume outside of the loop for randomized and non-randomized data
mean_randomized_data = {volume_file: np.mean(data_array, axis=0) for volume_file, data_array in randomized_data_arrays.items()}
#mean_non_randomized_data = {volume_file: np.mean(data_array, axis=0) for volume_file, data_array in non_randomized_data_arrays.items()}

'''
#---- Spatial correlations ----#

from neuromaps.datasets import fetch_annotation
from neuromaps.resampling import resample_images
from neuromaps.stats import compare_images

# Fetch desired annotation (add description, space, and density if needed for identification)
anno = fetch_annotation(source='hesse2017')

# Correlate with neuromaps annotation for randomized and non-randomized data
correlations_randomized = []
#correlations_non_randomized = []

for volume_file in volume_files:
    # Assuming 'mean_data' contains the spatial mean for each volume
    mean_data_randomized = mean_randomized_data[volume_file]
    #mean_data_non_randomized = mean_non_randomized_data[volume_file]

    # Resample data to match the annotation space
    data_resampled_randomized, anno_resampled_randomized = resample_images(src=combined_mask_img, trg=anno,
                                                                          src_space='MNI152', trg_space='MNI152',
                                                                          method='linear', resampling='downsample_only')

    #data_resampled_non_randomized, anno_resampled_non_randomized = resample_images(src=combined_mask_img, trg=anno,
    #                                                                              src_space='MNI152', trg_space='MNI152',
    #                                                                              method='linear', resampling='downsample_only')

    # Check mean values for each volume
    print(f"Mean value for {volume_file} - Randomized Data: {np.mean(mean_data_randomized)}")
    #print(f"Mean value for {volume_file} - Non-Randomized Data: {np.mean(mean_data_non_randomized)}")

    # Compare resampled data with neuromaps annotation using the compare_images function
    corr_randomized = compare_images(data_resampled_randomized, anno_resampled_randomized, metric='pearsonr')
    #corr_non_randomized = compare_images(data_resampled_non_randomized, anno_resampled_non_randomized, metric='pearsonr')

    # Append correlations to lists
    correlations_randomized.append(corr_randomized)
    #correlations_non_randomized.append(corr_non_randomized)

    # Print or use the correlation results as needed
    print(f"Correlation with Randomized Data for {volume_file}: {corr_randomized:.3f}")
    #print(f"Correlation with Non-Randomized Data for {volume_file}: {corr_non_randomized:.3f}")

# Plot the correlation values
# (Use the plotting code from the previous response)
'''


# TEST
# ... (existing code)

# Correlate with neuromaps annotation for randomized and non-randomized data
correlations_randomized = []
#correlations_non_randomized = []

for volume_file in volume_files:
    # Assuming 'mean_data' contains the spatial mean for each volume
    mean_data_randomized = mean_randomized_data[volume_file]
    #mean_data_non_randomized = mean_non_randomized_data[volume_file]

    # Create Nifti images from the mean data
    img_randomized = nib.Nifti1Image(mean_data_randomized, img.affine)
    #img_non_randomized = nib.Nifti1Image(mean_data_non_randomized, img.affine)

    # Resample data to match the annotation space using the randomized mask image
    data_resampled_randomized, anno_resampled_randomized = resample_images(src=img_randomized, trg=anno,
                                                                          src_space='MNI152', trg_space='MNI152',
                                                                          method='linear', resampling='downsample_only')
    # Check shapes for debugging
    print(f"Shape of resampled data (randomized): {data_resampled_randomized.shape}")
    print(f"Shape of resampled annotation (randomized): {anno_resampled_randomized.shape}")


    # Resample data to match the annotation space using the non-randomized mask image
    #data_resampled_non_randomized, anno_resampled_non_randomized = resample_images(src=img_non_randomized, trg=anno,
    #                                                                              src_space='MNI152', trg_space='MNI152',
    #                                                                              method='linear', resampling='downsample_only')

    #print(f"Shape of resampled data (non-randomized): {data_resampled_non_randomized.shape}")
    #print(f"Shape of resampled annotation (non-randomized): {anno_resampled_non_randomized.shape}")

    # ... (rest of the code)




#--- Plotting ---#

# Plotting the correlation values
plt.figure(figsize=(12, 6))

plt.plot(volume_files, correlations_randomized, label='Randomized Data', marker='o')
p#lt.plot(volume_files, correlations_non_randomized, label='Non-Randomized Data', marker='o')

# Adding labels and title
plt.xlabel('Volume')
plt.ylabel('Correlation Value')
plt.title('Correlation between Resampled Data and Neuromaps Annotation')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()



