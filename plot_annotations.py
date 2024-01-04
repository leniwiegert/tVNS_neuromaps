# Loading and displaying of the neuromaps annotations (brain maps)
# Author: Lena Wiegert

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import nilearn
from matplotlib.pyplot import colormaps
from nilearn import (plotting, datasets)
from nilearn import image as nli
from nilearn.regions import connected_regions
from nilearn.plotting import plot_roi


#----------- LOAD AND GET TO KNOW THE DATA ----------#

# The code atlas_fetching_correlation_test_file.py saves the desired annotation as .nii file on your computer
# Running the atlas_fetching code prior to this file is required

# Load data with its path
img = nib.load('/Users/leni/neuromaps-data/annotations/alarkurtti2015/raclopride/MNI152/source-alarkurtti2015_desc-raclopride_space-MNI152_res-3mm_feature.nii')

#----------- WORK WITH IMAGE DATA ----------#

# get_data() returns a standard numpy multidimensional array
img_data = img.get_fdata()

# Using a pre-defined brain mask (standard example brain for mango from the NAS)
mask_image = nib.load('/Users/leni/Documents/Master/Data/MNI152_T1_04mm_brain.nii')
mask_data = mask_image.get_fdata()


#----------- VISUALIZING THE AVERAGE SIGNAL OF A REGION ----------#

# ROI (region of interest) plot
# Keep only voxels with a higher value than 95% of all voxels
thr = nli.threshold_img(img, threshold='95%')
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
