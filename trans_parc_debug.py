
import os
import subprocess

import numpy as np
import nibabel as nib
from nibabel import save
from abagen import annot_to_gifti, relabel_gifti
from nilearn import image as nli
from nilearn import datasets, surface
from netneurotools import datasets as nntdata
from netneurotools.datasets import fetch_schaefer2018
from neuromaps import transforms, images, nulls, parcellate
from neuromaps.images import dlabel_to_gifti
import copy

#-- Debugging --#
import os
print(os.environ['PATH'])


#-------- LOAD AND PREP DATA --------#

# Define universal data directory
data_directory = '/home/leni/Documents/Master/data/'

img = nib.load(os.path.join(data_directory, '4D_rs_fCONF_del_taVNS_sham.nii'))

# Set numpy to print only 2 decimal digits for neatness
np.set_printoptions(precision=2, suppress=True)
# The array proxy allows us to create the image object without immediately loading all the array data from disk
nib.is_proxy(img.dataobj)

# Create mean image
mean_img = nli.mean_img(img)

# Replace non-finite values with a gray matter mask
gray_matter_mask_file = os.path.join(data_directory, 'out_GM_p_0_15.nii')
gray_matter_mask = nib.load(gray_matter_mask_file)

# Resample gray matter mask to match the resolution of mean_img
gray_matter_mask_resampled = nli.resample_to_img(gray_matter_mask, mean_img)

# Ensure both masks have the same shape
if not np.all(gray_matter_mask_resampled.shape == mean_img.shape):
    raise ValueError('Shape of input volume is incompatible.')

# Create a new mask by keeping only non-NaN values in both masks
mean_img_gm_data = np.where(np.isnan(mean_img.get_fdata()), gray_matter_mask_resampled.get_fdata(), mean_img.get_fdata())

# Create a new image with the new mask
mean_img_gm = nib.Nifti1Image(mean_img_gm_data.astype(np.float32), mean_img.affine)

# Make sure the affines are the same
mean_img_gm = nli.resample_to_img(mean_img_gm, img)

# Extract data from Nifti1Image
mean_img_data = mean_img_gm.get_fdata()


#-------- TRANSFORMATION --------#

# Transform MNI152 to fsLR (surface based data) with the density we want it in
# Nearest-neighbors  interpolation recommended for parcellation data
mean_img_fsav = transforms.mni152_to_fsaverage(mean_img_gm, '41k', method='nearest')
print(mean_img_fsav)
# The output is a tuple of Gifti Images

# Compare shapes of volumetric vs. fsLR/fsaverage data
print(mean_img_data.shape)
# (97, 115, 97)
print(images.load_data(mean_img_fsav).shape)
# (327684,), depending on the density you're in
mean_img_fsav_lh, mean_img_fsav_rh = mean_img_fsav
print(mean_img_fsav_lh.agg_data().shape)
# (163842,)

# Same for transformation to fsLR
mean_img_fslr = transforms.mni152_to_fslr(mean_img_gm, '32k', method='nearest')
print(mean_img_fslr)
# The output is a tuple of Gifti Images

# Check attributes of the left hemisphere image
print("Left Hemisphere Gifti Image:")
for array in mean_img_fsav_lh.darrays:
    print(array.data)

# Check attributes of the right hemisphere image
print("Right Hemisphere Gifti Image:")
for array in mean_img_fsav_rh.darrays:
    print(array.data)

# this part makes total sense, for fsLR and for fsaverage
# BUT: i need to make sure that mean_img_fsav correctly represents both hemispheres!
# it does :)


#-------- PARCELLATION --------#

# Parcellation as shown in the neuromaps doc:

from nilearn.datasets import fetch_atlas_schaefer_2018
from neuromaps.parcellate import Parcellater

# Fetch atlas with nilearn instead of netneurotools
#schaefer = fetch_atlas_schaefer_2018()

# Fetch atlas with netneurotools
schaefer = fetch_schaefer2018('fsaverage')['400Parcels7Networks']
schaefer_fsaverage = (schaefer)
#schaefer_fsaverage = relabel_gifti(schaefer_fsaverage)
print(schaefer_fsaverage)
# why doesn't he get the error?


def convert_annot_to_gifti(annot_file, output_gifti_file):
    # Load the annotation file
    annot = nib.freesurfer.read_annot(annot_file)

    # Create a GiftiImage object
    gifti = nib.gifti.GiftiImage()

    # Add annotation data to GiftiImage as meta data
    gifti.add_gifti_data_array(nib.gifti.GiftiDataArray(data=annot[0], intent='NIFTI_INTENT_LABEL'))

    # Save the GiftiImage to a file
    nib.save(gifti, output_gifti_file)

# Example usage
lh_annot_file = '/home/leni/nnt-data/atl-schaefer2018/fsaverage/atl-Schaefer2018_space-fsaverage_hemi-L_desc-400Parcels7Networks_deterministic.annot'
rh_annot_file = '/home/leni/nnt-data/atl-schaefer2018/fsaverage/atl-Schaefer2018_space-fsaverage_hemi-R_desc-400Parcels7Networks_deterministic.annot'

lh_output_gifti_file = '/home/leni/nnt-data/atl-schaefer2018/fsaverage/lh.Schaefer2018_400Parcels_7Networks.annot.gii'
rh_output_gifti_file = '/home/leni/nnt-data/atl-schaefer2018/fsaverage/rh.Schaefer2018_400Parcels_7Networks.annot.gii'

convert_annot_to_gifti(lh_annot_file, lh_output_gifti_file)
convert_annot_to_gifti(rh_annot_file, rh_output_gifti_file)

print(lh_output_gifti_file)
print(rh_output_gifti_file)


# Load the NIfTI image containing the maps data
#maps_img = nib.load(schaefer['maps'])

# Create a Parcellater instance using the duplicated maps data
#schaefer_parcellater = Parcellater(maps_img, space='fsaverage', hemi=None, resampling_target='parcellation')


#print(schaefer_parcellater)
'''
print(sorted(schaefer))
print(len(schaefer['maps']))
print(len(schaefer['labels']))
'''

# Check the shape of the fetched data
#print("Shape of fetched maps:", schaefer['maps'].shape)  # This will help us understand the structure
# Print out the keys of the fetched object
#print("Keys:", schaefer.keys())

# i have to relabel before this step

'''
# Parcellation with netneurotools
schaefer_fsav_41k = fetch_schaefer2018('fsaverage6')['400Parcels7Networks']
print(schaefer_fsav_41k)

# Obtain the file paths from the Surface object
lh_annotation_file = schaefer_fsav_41k.lh
rh_annotation_file = schaefer_fsav_41k.rh

# Convert annotation files to GIfTI images
lh_gifti_file = annot_to_gifti(lh_annotation_file)
rh_gifti_file = annot_to_gifti(rh_annotation_file)

# Load data from GIfTI files
lh_gifti_data = load_gifti(lh_gifti_file)
rh_gifti_data = load_gifti(rh_gifti_file)

# Create parcellater for left and right hemisphere separately
parc_fsav_lh = Parcellater(lh_output_gifti_file, 'fsaverage', resampling_target='parcellation', hemi='L')
parc_fsav_rh = Parcellater(rh_output_gifti_file, 'fsaverage', resampling_target='parcellation', hemi='R')

# Parcellate data for LH and RH separately
mean_img_fsav_lh_parc = parc_fsav_lh.fit_transform(mean_img_fsav_lh, 'fsaverage', hemi='L')
mean_img_fsav_rh_parc = parc_fsav_rh.fit_transform(mean_img_fsav_rh, 'fsaverage', hemi='R')
print(mean_img_fsav_lh_parc.shape)
print(mean_img_fsav_rh_parc.shape)
'''

schaefer_test = nntdata.fetch_schaefer2018('fslr32k')['400Parcels7Networks']
parc = Parcellater(dlabel_to_gifti(schaefer_test), 'fsLR')
print(schaefer_test)
print(parc)
mean_img_fslr_parc = parc.fit_transform(mean_img_fslr, 'fsLR')
print(mean_img_fslr_parc.shape)
#(400,)

#--------- SPATIAL NULL MODELS --------#

# Fetch annotation
#hesse2017 = datasets.fetch_annotation(source='hesse2017')

# Fetch the combined fsLR surfaces for both hemispheres
#fsLR_surfaces_combined = datasets.fetch_surf_fsaverage()
# Print the keys of the fetched dictionary
#print(fsLR_surfaces_combined.keys())

parcellation = ('/home/leni/nnt-data/atl-schaefer2018/fsaverage/lh.Schaefer2018_400Parcels_7Networks.annot.gii',
                '/home/leni/nnt-data/atl-schaefer2018/fsaverage/rh.Schaefer2018_400Parcels_7Networks.annot.gii')

'''

#---- DEBUGGING :) ----#

# Load parcellation files
parc_left = nib.load('/home/leni/nnt-data/atl-schaefer2018/fsaverage/lh.Schaefer2018_400Parcels_7Networks.annot.gii')
parc_right = nib.load('/home/leni/nnt-data/atl-schaefer2018/fsaverage/rh.Schaefer2018_400Parcels_7Networks.annot.gii')

# Output file paths for the surface mesh files
surf_left = '/home/leni/nnt-data/atl-schaefer2018/fsaverage/lh.Schaefer2018_400Parcels_7Networks.surf.gii'
surf_right = '/home/leni/nnt-data/atl-schaefer2018/fsaverage/rh.Schaefer2018_400Parcels_7Networks.surf.gii'

# Get the file paths for the surface mesh files
surf_left_path = '/home/leni/nnt-data/atl-schaefer2018/fsaverage/lh.Schaefer2018_400Parcels_7Networks.surf.gii'
surf_right_path = '/home/leni/nnt-data/atl-schaefer2018/fsaverage/rh.Schaefer2018_400Parcels_7Networks.surf.gii'

save(parc_left, surf_left)
save(parc_right, surf_right)

# Check if the surface mesh files are created successfully
if os.path.exists(surf_left) and os.path.exists(surf_right):
    print("Surface mesh files created successfully.")
else:
    print("Error creating surface mesh files.")

# Convert left hemisphere parcellation to surface mesh
os.system(f'mri_surf2surf --srcsubject fsaverage --hemi lh --sval {surf_left_path} --trgsubject fsaverage --tval {surf_left_path}')

# Convert right hemisphere parcellation to surface mesh
os.system(f'mri_surf2surf --srcsubject fsaverage --hemi rh --sval {surf_right_path} --trgsubject fsaverage --tval {surf_right_path}')

# Check if the surface mesh files are created successfully
if os.path.exists(surf_left) and os.path.exists(surf_right):
    print("Surface mesh files created successfully.")
else:
    print("Error creating surface mesh files.")

# Combine left and right hemisphere parcellation files
parcellation_combined = copy.deepcopy(parc_left)
parcellation_combined.darrays[0].data = np.concatenate((parc_left.darrays[0].data, parc_right.darrays[0].data))

# Output file path for the combined parcellation file
parcellation_combined_path = '/home/leni/nnt-data/atl-schaefer2018/fsaverage/Schaefer2018_400Parcels_7Networks_combined.annot.gii'

# Save the combined parcellation file
save(parcellation_combined, parcellation_combined_path)

# Convert the combined parcellation file to surface mesh
os.system(f'mri_surf2surf --srcsubject fsaverage --hemi lh --sval {parcellation_combined_path} --trgsubject fsaverage --tval {parcellation_combined_path}')
os.system(f'mri_surf2surf --srcsubject fsaverage --hemi rh --sval {parcellation_combined_path} --trgsubject fsaverage --tval {parcellation_combined_path}')

# Check if the combined parcellation file is created successfully
if os.path.exists(parcellation_combined_path):
    print("Combined parcellation file created successfully.")
else:
    print("Error creating combined parcellation file.")

'''
'''
# Get number of vertices
n_vertices_left_parc = parc_left.shape[0]  # Number of vertices in left hemisphere parcellation
n_vertices_right_parc = parc_right.shape[0]  # Number of vertices in right hemisphere parcellation

n_vertices_left_surf = surf_left[0].shape[0]  # Number of vertices in left hemisphere surface mesh
n_vertices_right_surf = surf_right[0].shape[0]  # Number of vertices in right hemisphere surface mesh

# Compare number of vertices
if n_vertices_left_parc == n_vertices_left_surf:
    print("Left hemisphere parcellation and surface mesh are compatible.")
else:
    print("Left hemisphere parcellation and surface mesh are not compatible.")

if n_vertices_right_parc == n_vertices_right_surf:
    print("Right hemisphere parcellation and surface mesh are compatible.")
else:
    print("Right hemisphere parcellation and surface mesh are not compatible.")'''

'''#-----------------#

# Generate null maps using the vasa method
# This function requires a single input file
mean_img_fslr_parc_nulls = nulls.vasa(data=mean_img_fslr_parc,
                                          atlas='fsLR',
                                          density='32k',
                                          parcellation=parcellation_combined_path,
                                          n_perm=10)

# Print the shape of the generated nulls
print(mean_img_fslr_parc_nulls.shape)

'''

# Calculate nulls with hungarian function
#mean_img_fslr_parc_nulls = nulls.hungarian(mean_img_fslr_parc, atlas='fsLR', density='32k', parcellation=parc, n_perm=10)
#mean_img_fslr_parc_nulls = nulls.hungarian(mean_img_fslr_parc, atlas='fsLR', density='32k', parcellation=parc, n_perm=10)

#print(mean_img_fslr_parc_nulls.shape)
# Error: 'str' object has no attribute 'agg_data'


# Parcellate hesse2017 before this next step
#corr, pval = stats.compare_images(mean_img_fsav_lh_parc, hesse2017_parc, nulls=rotated)
#print(f'r = {corr:.3f}, p = {pval:.3f}')
