# Author: Lena Wiegert

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import nilearn
from nilearn import plotting
from nilearn import image as nli
from nilearn.regions import connected_regions
from neuromaps.datasets import fetch_annotation
from neuromaps.resampling import resample_images
from nilearn.plotting import plot_roi
from matplotlib.colors import Normalize
from nilearn.image import resample_img
from neuromaps import stats
from neuromaps import datasets, images, nulls
from neuromaps.stats import compare_images

# Define universal data directory
data_directory = '/home/leni/Documents/Master/data/'

#----------- LOAD AND GET TO KNOW THE DATA ----------#

img = nib.load(os.path.join(data_directory, '4D_rs_fCONF_del_taVNS_sham.nii'))

# Check shape (x,y,z voxel dimensions + amount of volumes)
img.shape
# Set numpy to print only 2 decimal digits for neatness
np.set_printoptions(precision=2, suppress=True)
# The array proxy allows us to create the image object without immediately loading all the array data from disk
nib.is_proxy(img.dataobj)


#----------- WORK WITH IMAGE DATA ----------#

# get_data() returns a standard numpy multidimensional array
img_data_4d = img.get_fdata()

# Using a pre-defined brain mask (standard example brain for mango from the NAS)
mask_image = nib.load(os.path.join(data_directory, '4D_rs_fCONF_del_taVNS_sham.nii'))
mask_data_4d = mask_image.get_fdata()
# Double Check: There is brain activity (non-zero-values) in the array.

# Separate the volumes and create 41 NIfTI files
for volume_index in range(img_data_4d.shape[-1]):
    # Extract the volume at the specified index
    single_volume = img_data_4d[..., volume_index]
    # Create a new NIfTI image for the single volume
    single_volume_img = nib.Nifti1Image(single_volume, img.affine)
    # Save the single volume to a new NIfTI file
    output_path = f'/home/leni/Documents/Master/data/volume_{volume_index + 1}.nii'
    nib.save(single_volume_img, output_path)
    print(f"Volume {volume_index + 1} saved to {output_path}")

# Load all volumes
volumes = []
for i in range(1, 42):
    filename = os.path.join(data_directory, f'volume_{i}.nii')
    # Load the NIfTI file
    img = nib.load(filename)
    # Get the image data as a NumPy array
    data = img.get_fdata()
    # Append the data to the list
    volumes.append(data)

# Convert the list of volumes to a NumPy array
volumes_array = np.array(volumes)


#----------- IMAGE MANIPULATION - CREATE A MEAN IMAGE ----------#

# Mean image for all 41 volumes together
mean_img = nli.mean_img(img)

# Mean image for each volume separately
base_path = '/home/leni/Documents/Master/data/'
volume_files = [f'{base_path}volume_{i}.nii' for i in range(1, 42)]
# Initialize a list to store mean images
mean_images = []
# Loop through each volume file
for volume_file in volume_files:
    # Load the volume
    vol = nib.load(volume_file)
    # Get volume data
    vol_data = vol.get_fdata()
    # Calculate the mean image
    mean_img_vol = nli.mean_img(vol)
    # Assign a name to the mean image
    mean_img_name = f'mean_img_vol_{i + 1}'
    # Append the mean image to the list
    mean_images.append(mean_img_vol)
# Now, mean_images contains the mean images for each volume


#----------- VISUALIZING THE AVERAGE SIGNAL OF A REGION ----------#

# ROI (region of interest) plot
# Keep only voxels with a higher value than 95% of all voxels
# Change the following two lines, depending on which volume you want to display (index is one number lower)
mean_img_vol_1 = mean_images[0]
mean_img_vol_1_data = mean_img_vol_1.get_fdata()
# Double Check: There are non-NaN values in the array.

'''
# Threshold/Clustering not needed because of the gray matter mask

#thr = nli.threshold_img(mean_img_vol_1, threshold='95%')

# Keep the regions that are bigger than 1000mm^3
voxel_size = np.prod(thr.header['pixdim'][1:4])  # Size of 1 voxel in mm^3
print(voxel_size)

# Create a mask that only keeps those big clusters
cluster = connected_regions(thr, min_region_size=1000. / voxel_size, smoothing_fwhm=1)[0]
# Binarize the cluster file to create a overlay image
#overlay = nli.math_img('np.mean(img,axis=3) > 0', img=cluster)

# Handle non-finite values in the cluster data
cluster_data = cluster.get_fdata()
cluster_data[np.isnan(cluster_data)] = 0  # Replace NaN with 0

# Create a colormap mask based on the values in the cluster
cmap_mask_data = np.mean(cluster_data, axis=3)  # Adjust the threshold as needed
cmap_mask_img = nib.Nifti1Image(cmap_mask_data.astype(np.float32), cluster.affine)
'''


# Replace non-finite values with a gray matter mask
gray_matter_mask_file = os.path.join(data_directory, 'out_GM_p_0_15.nii')
gray_matter_mask = nib.load(gray_matter_mask_file)

# Choose either mean_img for all volumes or mean_img_vol_1-41 for the desired volume
cmap_mask_img = mean_img

# Resample gray matter mask to match the resolution of cmap_mask_img
gray_matter_mask_resampled = nli.resample_to_img(gray_matter_mask, cmap_mask_img)

# Ensure both masks have the same shape
if not np.all(gray_matter_mask_resampled.shape == cmap_mask_img.shape):
    raise ValueError('Shape of input volume is incompatible.')

# Create a new mask by keeping only non-NaN values in both masks
combined_mask_data = np.where(np.isnan(cmap_mask_img.get_fdata()), gray_matter_mask_resampled.get_fdata(), cmap_mask_img.get_fdata())

# Create a new image with the combined mask
combined_mask_img = nib.Nifti1Image(combined_mask_data.astype(np.float32), cmap_mask_img.affine)

# Necessary?
# Make sure the affines are the same
combined_mask_img = nli.resample_to_img(combined_mask_img, mask_image)

# Extract data from Nifti1Image
cmap_mask_img_data = combined_mask_img.get_fdata()
# Normalize the colorbar based on your data values
norm = Normalize(vmin=np.min(cmap_mask_img_data), vmax=np.max(cmap_mask_img_data))

'''
# Plotting, add background with colormap mask including gray matter
fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the figure size as needed (width, height)
plotting.plot_roi(
    combined_mask_img,
    bg_img=mask_image,
    display_mode='z',
    dim=-.5,
    cmap='hsv',
    cut_coords=None,    # Add this line to automatically choose the cut coordinates
    colorbar=True,
    axes=ax,
    vmin=np.min(cmap_mask_img_data),  # Set the vmin and vmax parameters
    vmax=np.max(cmap_mask_img_data)
)
plt.show()

'''
#------------ SPATIAL CORRELATIONS WITH THE MEAN IMAGE ------------#

# Test for mean_img_data and an example neuromaps annotation (here: NE receptor map hesse 2017)

# Fetch desired annotation (add description, space and density if needed for identification)
anno = fetch_annotation(source='hesse2017')

# Transformation with 'downsample_only’: The higher-resolution map is transformed to the space of the lower-resolution map
# Except if volumetric and non-volumetric space, then it's always transformed into the non-vol. space
data_res, anno_res = resample_images(src=combined_mask_img, trg=anno,
                                      src_space='MNI152', trg_space='MNI152',
                                      method='linear', resampling='downsample_only')

corr = compare_images(data_res, anno_res, metric='pearsonr')
print(f'The correlation value for the mean image of my data and the neuromaps annotation is r = {corr:.3f}.')

# The Pearson correlation coefficient ranges from -1 to 1, where:
# 1 indicates a perfect positive linear relationship,
# 0 indicates no linear relationship,
# -1 indicates a perfect negative linear relationship.


# --- Plotting ---#

# Plotting the correlations of NE, dopamine and serotonin PET maps with the mean image of my data

# Defining data with values and names
data = {
    'ding2010': -0.114,
    'hesse2017': -0.085,
    'alarkurtti2015': -0.088,
    'dukart2018': 0.017,
    'jaworska2020': -0.094,
    'kaller2017': -0.078,
    'sandiego2015': -0.089,
    'sasaki2012': -0.091,
    'smith2017': -0.085,
    'fazio2016': -0.030,
    'gallezot2010': -0.133,
    'radnakrishnan2018': -0.107,
    'salvi2012_1': 0.023,
    'salvi2012_2': -0.019,
    'salvi2012_3': -0.085,
    'salvi2012_4': 0.112
}

# Extract names and values from the data dictionary
annos = list(data.keys())
values = list(data.values())

# Define categories for each name
categories = {
    'NE PET': ['ding2010', 'hesse2017'],
    'Dopamine PET': ['alarkurtti2015', 'dukart2018', 'jaworska2020', 'kaller2017', 'sandiego2015', 'sasaki2012', 'smith2017'],
    'Serotonin PET': ['fazio2016', 'gallezot2010', 'radnakrishnan2018', 'salvi2012_1', 'salvi2012_2', 'salvi2012_3', 'salvi2012_4']
}

# Define colors for each category
category_colors = {
    'NE PET': 'lightcoral',
    'Dopamine PET': 'lightgreen',
    'Serotonin PET': 'lightblue'
}

# Plot the data with shaded background for each category
for category, names_in_category in categories.items():
    plt.axvspan(annos.index(names_in_category[0]) - 0.5, annos.index(names_in_category[-1]) + 0.5,
                facecolor=category_colors[category], alpha=0.5)

plt.scatter(annos, values, color='black', s=30, zorder=10)
plt.axhline(0, color='black', linestyle='dashed', linewidth=1)  # Add a horizontal line at y=0
plt.xlabel('Annotation')
plt.ylabel('Pearson correlation coefficient')
plt.title('Spatial correlation of the tVNS data (mean image) and NE, dopamine, and serotonin PET maps')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()



#------------ SPATIAL CORRELATIONS FOR ALL 41 FILES ------------#

# Each volume of my data compared to Hesse 2017

correlation_values = []

for i in range(1, 42):
    filename_mean = f'/Users/leni/Documents/Master/Data/volume_{i}.nii'
    filename_combined_mask = f'combined_mask_img_{i}.nii'
    mean_img_vol = nib.load(filename_mean)

    # Additional code to define combined_mask_data_1 to combined_mask_data_41
    gray_matter_mask_resampled = nilearn.image.resample_to_img(gray_matter_mask, mean_img_vol)

    if not np.all(gray_matter_mask_resampled.shape == mean_img_vol.shape):
        raise ValueError('Shape of input volume is incompatible.')

    combined_mask_data = np.where(np.isnan(mean_img_vol.get_fdata()), gray_matter_mask_resampled.get_fdata(), mean_img_vol.get_fdata())

    combined_mask_img = nib.Nifti1Image(combined_mask_data.astype(np.float32), mean_img_vol.affine)
    combined_mask_img = nilearn.image.resample_to_img(combined_mask_img, mask_image)

    combined_mask_img_data = combined_mask_img.get_fdata()
    norm = Normalize(vmin=np.min(combined_mask_img_data), vmax=np.max(combined_mask_img_data))

    # Continue with the rest of the code as in the previous example
    # Fetch annotation
    hesse2017 = fetch_annotation(source='hesse2017')

    # Resample the second image to match the dimensions of the first image
    img_hesse2017_resampled = nilearn.image.resample_img(hesse2017, target_affine=combined_mask_img.affine,
                                                         target_shape=combined_mask_img.shape,
                                                         interpolation='nearest')

    data_res, hesse_res = resample_images(src=combined_mask_img, trg=hesse2017,
                                          src_space='MNI152', trg_space='MNI152',
                                          method='linear', resampling='downsample_only')

    # Extract data arrays
    data_hesse2017_rs = img_hesse2017_resampled.get_fdata()

    corr = compare_images(data_res, hesse_res, metric='pearsonr')
    correlation_values.append(corr)

    print(f"Processing {filename_combined_mask}")
    print("Original lengths - data_combined_mask:", len(combined_mask_img_data), "data_hesse2017:", len(data_hesse2017_rs))
    print(f'r = {corr:.3f}')
    print("\n")

# Print the summary array of correlation values
print("Here are the spatial correlations for my data with Hesse 2017:")
print(np.array(correlation_values))


# --- Plotting ---#

# Assuming correlation_values is your array of correlation values
correlation_values = np.array([-0.089, 0.057, 0.174, 0.176, -0.203, 0.141, -0.109, 0.127, -0.039, 0.021, -0.057, -0.172, 0.191, 0.198, 0.252, -0.051, -0.029, 0.224, 0.183, -0.102, 0.054, 0.191, -0.043, 0.203, -0.095, -0.094, -0.215, 0.011, -0.315, 0.073, -0.049, 0.047, -0.149, 0.429, 0.114, -0.065, -0.092, -0.146, 0.127, -0.122, -0.085])

# Generate a range from 1 to 41
x_values = np.arange(1, 42)

# Plot the values
plt.plot(x_values, correlation_values, marker='o', linestyle='-')
plt.title('Spatial Correlation Values')
plt.xlabel('Volume Index')
plt.ylabel('Correlation Coefficient')
plt.grid(True)
plt.show()







