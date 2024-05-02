
import os
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from nilearn import image as nli
from brainsmash.workbench.geo import volume
from brainsmash.mapgen.eval import sampled_fit
from brainsmash.mapgen.sampled import Sampled



# Define universal data directory
data_directory = '/home/leni/Documents/Master/data/'
#data_directory = '/home/neuromadlab/tVNS_project/data/'
output_directory = '/home/leni/Documents/Master/data/'

annotation_sources = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

'''
# Load 3D tVNS images (whole-brain, unparcellated)
wholebrain_img = nib.load(os.path.join(data_directory, '4D_rs_fCONF_del_taVNS_sham.nii'))

# Replace non-finite values with a gray matter mask
gray_matter_mask_file = os.path.join(data_directory, 'out_GM_p_0_15.nii')
gray_matter_mask = nib.load(gray_matter_mask_file)

# Resample gray matter mask to match the resolution of mean_img
gray_matter_mask_resampled = nli.resample_to_img(gray_matter_mask, wholebrain_img)

# Process each volume
for i in range(1, 42):
    vol_path = os.path.join(data_directory, f'volume_{i}.nii')
    vol = nib.load(vol_path)
    # Resample and add GM mask
    gray_matter_mask_resampled = nli.resample_to_img(gray_matter_mask, vol)
    # Create a new mask by keeping only non-NaN values in both masks
    vol_gm_data = np.where(np.isnan(vol.get_fdata()), gray_matter_mask_resampled.get_fdata(), vol.get_fdata())
    # Create a new image with the new mask
    vol_gm = nib.Nifti1Image(vol_gm_data.astype(np.float32), vol.affine)
    # Save the masked data
    # nib.save(vol_1_gm, os.path.join(data_directory, 'vol_1_gm.nii'))
    # Extract affine transformation matrix
    affine = vol_gm.affine
    # Get voxel coordinates
    coords = np.array(np.meshgrid(*[range(d) for d in vol_gm_data.shape], indexing='ij')).reshape(3, -1).T
    # Save voxel coordinates to a text file containing voxel coordinates (with shape N rows by 3 columns)
    coords_filename = os.path.join(output_directory, f'vol_{i}_voxel_coordinates.txt')
    #np.savetxt(coords_filename, coords, fmt='%d')
    # Flatten data array
    values = vol_gm_data.flatten()
    # Save brain map values to a text file containing N brain map values
    values_filename = os.path.join(output_directory, f'vol_{i}_brain_map_values.txt')
    #np.savetxt(values_filename, values)

'''


# Brainsmash #

coord_file = os.path.join(data_directory, 'vol_1_voxel_coordinates.txt')
brain_map = os.path.join(data_directory, 'vol_1_brain_map_values.txt')

filenames = volume(coord_file, data_directory) #data_directory = output_directory
print(f'The memory-mappes distance matrix files are saved: {filenames}')

# These are three of the key parameters affecting the variogram fit
kwargs = {'ns': 500,
          'knn': 1500,
          'pv': 70
          }

# Process was terminated by the operating system using the SIGKILL signal, possibly because it exceeded resource limits (such as memory usage) or because it took too long to complete.

'''
# Running this command will generate a matplotlib figure
sampled_fit(brain_map, filenames['D'], filenames['index'], nsurr=10, **kwargs)

gen = Sampled(x=brain_map, D=filenames['D'], index=filenames['index'], **kwargs)
surrogate_maps = gen(n=2) # n=1000
'''

