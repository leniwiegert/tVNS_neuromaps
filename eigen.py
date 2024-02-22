# Install eigenstrapping package from GitHub
# git clone https://github.com/SNG-newy/eigenstrapping.git
# cd eigenstrapping
# python3 -m pip install .

'''
Quick introduction to brain maps and eigenmodes
===============================================

Eigenmodes of a surface encode all pairwise (auto)correlations (i.e., smoothness).
Another property of eigenmodes: they are orthogonal. By taking random rotations of them, one can
create new brain maps with the same smoothness but randomized topology.
'''


#----------- EIGENSTRAPPING EXAMPLE -----------#
import os
import numpy as np
import nibabel as nib
from nilearn import image as nli
import matplotlib.pyplot as plt
from tqdm import trange
from nilearn import datasets, surface
from netneurotools import datasets as nntdata
from netneurotools.datasets import fetch_schaefer2018
from neuromaps import transforms, images, nulls, parcellate
from neuromaps import datasets
from neuromaps.nulls import alexander_bloch
from neuromaps.parcellate import Parcellater
from neuromaps.datasets import fetch_annotation
from neuromaps.images import (construct_shape_gii, load_data, annot_to_gifti,
                              relabel_gifti, dlabel_to_gifti)


from eigenstrapping.datasets import load_surface_examples
from eigenstrapping.plotting import csplot
from eigenstrapping import SurfaceEigenstrapping
from eigenstrapping import datasets
from eigenstrapping import utils
from eigenstrapping.utils import get_schaefer


surf_lh, surf_rh, data_lh, data_rh, emodes_lh, emodes_rh, evals_lh, evals_rh = load_surface_examples(with_surface=True)
print(surf_lh)
print(data_lh.shape)
print(emodes_lh.shape)
print(evals_lh.shape)

csplot(data_lh, 'fsaverage')

eigen = SurfaceEigenstrapping(
                data=data_lh,
                emodes=emodes_lh,
                evals=evals_lh,
                num_modes=100,
                resample=True,
                )

surr = eigen.generate()
csplot(surr, 'fsaverage')

# neuromaps: Version: 0.0.5
# eigenstrapping: Version: 0.0.1.10

#------------------------------------------------------------




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
print(mean_img_gm.shape)

# Extract data from Nifti1Image
mean_img_data = mean_img_gm.get_fdata()
print(mean_img_data.shape)

# Split mean image in left and right hem

# Assume midline
data_array = mean_img_gm.get_fdata()
midline_index = mean_img_gm.shape[2] // 2  # The // operator is used for integer division
left_hemisphere = data_array[:, :, :midline_index]
right_hemisphere = data_array[:, :, midline_index:]

left_img = nib.Nifti1Image(left_hemisphere, img.affine)
right_img = nib.Nifti1Image(right_hemisphere, img.affine)

nib.save(left_img, '/home/leni/Documents/Master/data/left_hemisphere.nii')
nib.save(right_img, '/home/leni/Documents/Master/data/right_hemisphere.nii')

# Transpose the array if needed
left_img_data = left_img.get_fdata()
left_img_data_T = left_img_data.transpose()

# Create a new Nifti1Image object with the transposed data
left_img_T = nib.Nifti1Image(left_img_data_T, left_img.affine)

# repeat for rh


#---------- NULLS WITH PARCELLATED DATA ---------#

# Try with my parcellated data
schaefer = utils.get_schaefer()
print(schaefer)


# Try with left hem
parcellation = schaefer[0] # lh
mean_img_parc = utils.calc_parcellate(parcellation, left_img)
#genepc_parc = utils.calc_parcellate(parcellation, genepc_lh)
#print(gradient_parc.shape, genepc_parc.shape)
#(200,) (200,)



'''
Traceback (most recent call last):
  File "/home/leni/PycharmProjects/tVNS_neuromaps/eigen.py", line 144, in <module>
    mean_img_parc = utils.calc_parcellate(parcellation, left_img)
  File "/home/leni/.local/lib/python3.10/site-packages/eigenstrapping/utils.py", line 297, in calc_parcellate
    data_input = data_input.T
AttributeError: 'Nifti1Image' object has no attribute 'T'
'''

# currently working on splitting the volumetric data in 2 hems correctly




