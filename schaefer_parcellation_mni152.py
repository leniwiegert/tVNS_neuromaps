import os
from neuromaps import nulls
from neuromaps.parcellate import Parcellater
from nilearn.datasets import fetch_atlas_schaefer_2018
import numpy as np
from neuromaps.datasets import fetch_annotation
from neuromaps.stats import compare_images
from nilearn import image as nli
import nibabel as nib
from tqdm import tqdm


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


#--------- PARCELLATION --------#

# Trying the same with nilearn as in the Neuromaps Workshop Video
schaefer = fetch_atlas_schaefer_2018(n_rois=400)
print(schaefer['maps'])
parc = Parcellater(schaefer['maps'], space='MNI152')
print(parc)
mean_img_parc = parc.fit_transform(mean_img_gm, 'MNI152')
print(mean_img_parc.shape)
# Shape should be (1, 400)
print(mean_img_parc)

# Fetch annotation
anno = fetch_annotation(source='hesse2017')
# Parcellate the annotation we want to compare to our data
anno_parc = parc.fit_transform(anno, 'MNI152')
print(anno_parc.shape)


#-------- NULL MODELS --------#

# Compare resampled original data with neuromaps annotation using the compare_images function
corr_original = compare_images(mean_img_parc, anno_parc, metric='pearsonr')
# Print the correlation result as needed
print(f'Correlation of the Annotation and the Mean Image: {corr_original}')

# Calculate nulls for volumetric data with status bar

for _ in tqdm(range(3)):
    nulls = nulls.moran(mean_img_parc, atlas='MNI152', parcellation=schaefer['maps'], density='2mm', n_perm=10, seed=1234)
print(nulls.shape)


