
import os
import subprocess

import numpy as np
import nibabel as nib
from nibabel import save
from abagen import annot_to_gifti, relabel_gifti
from nilearn import image as nli
from nilearn import datasets, surface
from netneurotools import datasets as nntdata
from neuromaps.datasets import fetch_annotation
from neuromaps import transforms, images, nulls, parcellate, stats
from neuromaps.images import dlabel_to_gifti
from neuromaps.parcellate import Parcellater
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.datasets import fetch_atlas_surf_destrieux

#-- Debugging --#
import os
print(os.environ['PATH'])
# Add the path to the wb_command executable to the PATH environment variable
os.environ["PATH"] += os.pathsep + "/opt/workbench/bin_linux64"


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
mean_img_fsav = transforms.mni152_to_fsaverage(mean_img_gm, '10k', method='nearest')
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

schaefer = nntdata.fetch_schaefer2018('fslr32k')['400Parcels7Networks']
parc = Parcellater(dlabel_to_gifti(schaefer), 'fsLR')
mean_img_fslr_parc = parc.fit_transform(mean_img_fslr, 'fsLR')
print(mean_img_fslr_parc)


#--------- NULL MODELS ---------#

# Nulls models as in the neuromaps documentation (with Destrieux)

hesse2017 = fetch_annotation(source='hesse2017')
smith2017 = fetch_annotation(source='smith2017')
abagen = fetch_annotation(source='abagen')
abagen_parc = parc.fit_transform(abagen, 'fsaverage')

destrieux = fetch_atlas_surf_destrieux()
print(sorted(destrieux))
print(len(destrieux['map_left']), len(destrieux['map_right']))
print(len(destrieux['labels']))

labels = [label.decode() for label in destrieux['labels']]
parc_left = images.construct_shape_gii(destrieux['map_left'], labels=labels,
                                       intent='NIFTI_INTENT_LABEL')
parc_right = images.construct_shape_gii(destrieux['map_right'], labels=labels,
                                        intent='NIFTI_INTENT_LABEL')
parcellation = images.relabel_gifti((parc_left, parc_right), background=['Medial_wall'])
print(parcellation)

# Destrieux is for the 10k fsaverage coordinate system
from neuromaps import parcellate
destrieux = parcellate.Parcellater(parcellation, 'fsaverage').fit()
mean_img_fsav_parc = destrieux.transform(mean_img_fsav, 'fsaverage')
abagen_parc = destrieux.transform(abagen, 'fsaverage')
mean_img_fsav_parc_nulls = nulls.alexander_bloch(mean_img_fsav_parc, atlas='fsaverage', density='10k', n_perm=100, seed=1234, parcellation=parcellation)
print(mean_img_fsav_parc_nulls.shape)

corr, pval = stats.compare_images(mean_img_fsav_parc, abagen_parc, nulls=mean_img_fsav_parc_nulls)
print(f'r = {corr:.3f}, p = {pval:.3f}')



#--------- NULL MODELS WITH SCHAEFER/HUNGARIAN ---------#

# Parcellation as shown in the neuromaps doc:

schaefer = nntdata.fetch_schaefer2018('fslr32k')['400Parcels7Networks']
print(schaefer)
parc = Parcellater(dlabel_to_gifti(schaefer), 'fsLR')
print(parc)
mean_img_fslr_parc = parc.fit_transform(mean_img_fslr, 'fsLR')
print(mean_img_fslr_parc.shape)
# The Parcellater needs to be a tuple of strings in order to use it for the nulls functions
print(dir(parc))

# Trying the same with nilearn as in the Neuromaps Workshop Video
schaefer_nli = fetch_atlas_schaefer_2018(n_rois=400)
print(schaefer_nli['maps'])
parc_nli = Parcellater(schaefer_nli['maps'], space='fsLR')
print(parc_nli)
mean_img_fslr_parc_nli = parc_nli.fit_transform(mean_img_fslr, 'fsLR')
print(mean_img_fslr_parc_nli.shape)
# Shape should be (1, 400), not (400,)
mean_img_fslr_parc_nli_resh = mean_img_fslr_parc_nli.reshape(1, -1)
#print(mean_img_fslr_parc_nli_resh)


# Calculate nulls with hungarian function
mean_img_fslr_parc_nulls = nulls.hungarian(mean_img_fslr_parc, atlas='fsLR', density='32k', parcellation=parc, n_perm=10)
print(mean_img_fslr_parc_nulls.shape)
# Error: 'str' object has no attribute 'agg_data'



# Problem: Bei der Parcellierung von Hesse/Smith kommt immer 'must specify hemi', egal ob mit Schaefer oder mit Destrieux

# Problem: mean_img_fslr_parc_nli.shape ist (400,), bei Justine im Video aber (1, 400)
# Reshaping?
# Problem before that: hemi must be defined for line 165



