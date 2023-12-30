# Loading and displaying of an fMRI dataset of 41 patients
# Author: Lena Wiegert

import os
import pytest
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import nilearn
from matplotlib.pyplot import colormaps
from nilearn import (plotting, datasets)
from nilearn import image as nli
from nibabel.testing import data_path
from nilearn.regions import connected_regions
from nilearn.plotting import plot_roi
from nilearn.masking import apply_mask
from nilearn.masking import unmask


#----------- LOAD AND GET TO KNOW THE DATA ----------#

img = nib.load('/Users/leni/Documents/Master/Data/4D_rs_fCONF_del_taVNS_sham.nii')

# Check shape (x,y,z voxel dimensions + amount of volumes)
img.shape
# Set numpy to print only 2 decimal digits for neatness
np.set_printoptions(precision=2, suppress=True)
# NIfTI images have an affine relating the voxel coordinates to world coordinates in RAS+ space
img.affine
# Data type (float32)
img.get_data_dtype()
# The header contains the metadata for this image
img.header
# The array proxy allows us to create the image object without immediately loading all the array data from disk
nib.is_proxy(img.dataobj)


#----------- WORK WITH IMAGE DATA ----------#

# get_data() returns a standard numpy multidimensional array
img_data_4d = img.get_fdata()

# Using a pre-defined brain mask (standard example brain for mango from the NAS)
mask_image = nib.load('/Users/leni/Documents/Master/Data/MNI152_T1_04mm_brain.nii')
mask_data_4d = mask_image.get_fdata()

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

'''
# not finished yet
#----------- CREATE A STATIC 3D PLOT ----------#

# Extract one volume (e.g. the first volume)
# vol_to_display = img.slicer[..., 0]
# Display a static 3D image
vol_1 = nib.load('/Users/leni/Documents/Master/Data/volume_1.nii')
plotting.plot_stat_map(vol_1, threshold=20)
# Display an interactive plot (3D), doesn't work yet
# plotting.view_img(vol_1,threshold='auto')
'''

#----------- VISUALIZING THE AVERAGE SIGNAL OF A REGION ----------#

# ROI (region of interest) plot
# Keep only voxels with a higher value than 95% of all voxels
# Change the following two lines, depending on which volume you want to display (index is one number lower)
mean_img_vol_1 = mean_images[0]
thr = nli.threshold_img(mean_img_vol_1, threshold='95%')
# Keep the regions that are bigger than 1000mm^3
voxel_size = np.prod(thr.header['pixdim'][1:4])  # Size of 1 voxel in mm^3
print(voxel_size)
# Create a mask that only keeps those big clusters
cluster = connected_regions(thr, min_region_size=1000. / voxel_size, smoothing_fwhm=1)[0]
# Binarize the cluster file to create a overlay image
overlay = nli.math_img('np.mean(img,axis=3) > 0', img=cluster)
# Plotting, add background
plotting.plot_roi(overlay, bg_img=mask_image, display_mode='z', dim=-.5, cmap='plasma');
plotting.show()




