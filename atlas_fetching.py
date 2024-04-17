# Fetching atlases and using spatial null models (test file)
# Author: Lena Wiegert

'''
This code provides an overview on how to fetch brain atlases (annotations, brain maps) from neuromaps.
In addition to that, it shows the use of spatial null models to test the statistical correlation between two
brain maps.
'''

#--------------- FETCHING ATLASES ----------------#

# example for fetching atlases from neuromaps:
import neuromaps as nm
from neuromaps import datasets

# Currently available Neuromaps annotations
annotations = datasets.available_annotations()
print(f'Available annotations: {len(annotations)}')

# Example: Download of PET tracer binding (BPnd) to D2 (dopamine receptor) from a 2015 study
alarkurtti = datasets.fetch_annotation(source='alarkurtti2015', desc='raclopride')
print(alarkurtti)
# The NIFTI file of this annotation can now be found in the corresponding folder!




'''
# not in use
#--------------- USING SPATIAL NULL MODELS ----------------#

# How to use spatial null models in neuromaps.nulls to test the correlation between two brain annotations
# Spatial null models need to be used whenever you’re comparing brain maps

# Example with two annotations
from neuromaps import datasets
ding = datasets.fetch_annotation(source='ding2010', return_single=True)
abagen = datasets.fetch_annotation(source='abagen', return_single=True)
print('Ding2010: ', ding)
print('Abagen: ', abagen)

# These annotations are in different spaces, so we first need to resample them to the same space
# The data returned will always be pre-loaded nibabel image instances
from neuromaps import resampling
ding, abagen = resampling.resample_images(src=ding, trg=abagen,
                                            src_space='MNI152',
                                            trg_space='fsaverage',
                                            resampling='transform_to_alt',
                                            alt_spec=('fsaverage', '10k'))
print(ding, abagen)

# Correlation of the resampled images
from neuromaps import stats
corr = stats.compare_images(ding, abagen)
print(f'Correlation: r = {corr:.02f}')


# Statistical significance of this correlation
from neuromaps import images, nulls
ding_data = images.load_data(ding)
rotated = nulls.alexander_bloch(ding_data, atlas='fsaverage', density='10k',
                                n_perm=100, seed=1234)
print(rotated.shape)

corr, pval = stats.compare_images(ding, abagen, nulls=rotated)
print(f'Correlation: r = {corr:.02f}, p = {pval:.04f}')
'''

