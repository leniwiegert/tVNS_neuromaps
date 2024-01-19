# Author: Lena Wiegert

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
from scipy.stats import pearsonr


#----------- LOAD AND GET TO KNOW THE DATA ----------#

img = nib.load('/Users/leni/Documents/Master/Data/4D_rs_fCONF_del_taVNS_sham.nii')

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
mask_image = nib.load('/Users/leni/Documents/Master/Data/MNI152_T1_04mm_brain.nii')
mask_data_4d = mask_image.get_fdata()
# Double Check: There is brain activity (non-zero-values) in the array.

# Separate the volumes and create 41 NIfTI files
for volume_index in range(img_data_4d.shape[-1]):
    # Extract the volume at the specified index
    single_volume = img_data_4d[..., volume_index]
    # Create a new NIfTI image for the single volume
    single_volume_img = nib.Nifti1Image(single_volume, img.affine)
    # Save the single volume to a new NIfTI file
    output_path = f'/Users/leni/Documents/Master/Data/volume_{volume_index + 1}.nii'
    nib.save(single_volume_img, output_path)
    print(f"Volume {volume_index + 1} saved to {output_path}")

# Load all volumes
volumes = []
for i in range(1, 42):
    filename = f'/Users/leni/Documents/Master/Data/volume_{i}.nii'
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
base_path = '/Users/leni/Documents/Master/Data/'
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
#why this threshold tho?
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
gray_matter_mask_file = '/Users/leni/Documents/Master/Data/out_GM_p_0_15.nii'  # Provide the path to your gray matter mask
gray_matter_mask = nib.load(gray_matter_mask_file)

cmap_mask_img = mean_img_vol_1

# Resample gray matter mask to match the resolution of cmap_mask_img
gray_matter_mask_resampled = nli.resample_to_img(gray_matter_mask, cmap_mask_img)

# Ensure both masks have the same shape
if not np.all(gray_matter_mask_resampled.shape == cmap_mask_img.shape):
    raise ValueError('Shape of input volume is incompatible.')

# Create a new mask by keeping only non-zero voxels in both masks
#combined_mask_data = np.logical_and(cmap_mask_img.get_fdata() > 0, gray_matter_mask_resampled.get_fdata() > 0)

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


#------------ SPATIAL CORRELATIONS ------------#
# Test for combined_mask_data and an example neuromaps annotation (here: NE receptor map)

# Fetch annotation
hesse2017 = fetch_annotation(source='hesse2017')

# Resample the second image to match the dimensions of the first image
img_hesse2017_resampled = resample_img(hesse2017, target_affine=combined_mask_img.affine,
                                       target_shape=combined_mask_img.shape,
                                       interpolation='nearest')

data_res, hesse_res = resample_images(src=combined_mask_img, trg=hesse2017,
                                      src_space='MNI152', trg_space='MNI152',
                                      method='linear', resampling='downsample_only')

# Extract data arrays
#data_combined_mask = data_res
data_hesse2017_rs = img_hesse2017_resampled.get_fdata()

# Check map lengths
print("Original lengths - data_combined_mask:", len(cmap_mask_img_data), "data_hesse2017:", len(data_hesse2017_rs))

# Flatten the arrays while maintaining spatial correspondence
data_combined_mask_flattened = cmap_mask_img_data.flatten()
data_hesse2017_flattened = data_hesse2017_rs.flatten()

# Calculate spatial correlation using Pearson correlation coefficient
corr_coeff, _ = pearsonr(data_combined_mask_flattened, data_hesse2017_flattened)
print(f"Spatial Correlation: {corr_coeff:.2f}")

# The Pearson correlation coefficient ranges from -1 to 1, where:
# 1 indicates a perfect positive linear relationship,
# 0 indicates no linear relationship,
# -1 indicates a perfect negative linear relationship.

# Questions:
# Is flattening to a 1D array okay if it maintains the spatial correspondence between two arrays?
# What about the threshold/clustering? Necessary?


