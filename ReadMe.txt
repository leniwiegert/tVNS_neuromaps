
Placing the Effects of Transcutaneous Vagus Nerve Stimulation into Neurobiological Context 

 -----------------------------------------------------------------------------------

- Table of Contents -

1. Project Description
2. Installation and Set Up
	2.1 Toolbox Installation (neuromaps_first_steps.py)
	2.2 Downloads
3. User Guide
	3.1 Atlas Fetching and Simple Correlations (atlas_fetching.py)
	3.2 Data Loading, ROI Plotting and Correlation Calculation (Group Level - Whole-Brain - 
	    No Parcellation - sc_wholebrain_group.py)
 	3.3 Randomization of the tVNS effect (permutation_tVNS_effect.py)
 	3.4 Bootstrapped Spatial Correlations and Null Models of the rs-FC Maps of tVNS-induced Changes with PET 
 	    Receptor Maps (Group + Ind. Level - Cortical - Parcellation with Schaefer - 
 	    sc_groupind_parc_cortical.py)
 	3.5 Comparison of Cortical and Whole-Brain Correlations (Group + Ind. Level - Parcellation
 	    with Schaefer and Tian - sc_groupind_parc_wholebrain.py)
  	3.6 Raincloud Plotting (raincloudplots.py)
  	3.7 Spatial Nulls of Whole-Brain Images (brainsmash) 
  4. Credits
  
  
  
 -----------------------------------------------------------------------------------
 
- Code Overview -

1. Project Description:
	
This project investigates the neurobiological mechanisms underlying the effects of non-invasive transcutaneous Vagus Nerve Stimulation (tVNS). Using functional magnetic resonance imaging (fMRI) data and advanced analytical techniques, the study aims to uncover associations between tVNS-induced brain responses and neurobiological factors, such as receptor density patterns. By exploring individual patterns of brain response to tVNS compared to group-level data, we aim to elucidate heterogeneity in tVNS responses and seek to advance our understanding of tVNS as a potential therapeutic intervention.
The code for this project has been written in Python and uses several neuroscientific toolboxes, with a focus on a recently developed open-source toolbox called Neuromaps. Overall, the code aims to statistically compare the resting-state functional connectivity brain maps of tVNS-induced changes (rs-FC maps) to PET receptor density maps on individual and group level. 

 -----------------------------------------------

2. Installation and Set Up:
2.1 Toolbox Installation (neuromaps_first_steps.py)
- Download connectome workbench: 
  https://www.humanconnectome.org/software/get-connectome-workbench)
- Create environment and download neuromaps by following the installation steps in the documentation: 	
  https://neuromaps.readthedocs.io/en/latest/installation.html
2.2 Downloads
- Download the following data files from the NAS:
	- rs-FC maps of tVNS-induced changes (4D_rs_fCONF_del_taVNS_sham.nii)
	- Group level image with GM mask (combined_mask.nii.gz)
	- Individual level images without GM mask (volume_1.nii to volume_41.nii)
	
 -----------------------------------------------
	
3. User Guide:
3.1 Atlas Fetching and Simple Correlations (atlas_fetching.py)
	- Check out currently available files of Neuromaps 
	- Example for D2 receptor map
3.2 Data Loading, ROI Plotting and Correlation Calculation (sc_wholebrain_group.py)
	- Load tVNS data and check out its properties (header, affine, ...)
	- Create and load seperate files for single subject data (individual data of each participant, 
	  named by their ID)
	- Create mean image of individual data files
	- Try different plotting options 
	- Calculate spatial correlations of the rs-FC maps of tVNS-induced changes with 11 PET receptor maps 
	  on group and individual level
 3.3 Randomization of the tVNS effect (permutation_tVNS_effect.py) 
 3.4 Bootstrapped Spatial Correlations and Null Models of the rs-FC Maps of tVNS-induced Changes with PET Receptor Maps 
 (Group + Ind. Level - Cortical - Parcellation with Schaefer - sc_groupind_parc_cortical.py)
	 - Load and prep data as previously shown 
	 - Transform data and annotation from MNI152 (volumetric) to fsLR (surface-based) space
	- Parcellate data and annotation with Schaefer2018 (Cortical Brain Atlas) 
	- Group Level: Calculate nulls and spatial correlations for the rs-FC maps of tVNS-induced changes with 
	  11 PET receptor maps 
	- Individual Level: Same as on group level
	- Heatmap plotting on group and individual level
3.5 Comparison of Cortical and Whole-Brain Correlations 
(Group + Ind. Level - Parcellation with Schaefer and Tian - sc_groupind_parc_wholebrain.py)
	- Download Melbourne Subcortical Brain Atlas (Ye Tian): 
	https://www.nitrc.org/frs/download.php/13364/Tian2020MSA_v1.4.zip
	- Define your atlas_directory (line 217)
	- Load and prep data as previously shown 
	- Cortical data (group + individual):
		- Transformation and parcellation of cortical images with Schaefer (load data instead and remove 
	 	  parcellation part)
		- Calculation of the spatial correlations with 11 PET receptor maps 
	- Whole-brain images (group + individual):
		- Parcellation of whole-brain images with Tian (Melbourne Subcortical Brain Atlas) 
		- Calculation of the spatial correlations with 11 PET receptor maps 
	- Plotting for comparison of cortical and whole-brain data
3.6 Still open - Raincloud Plotting (raincloudplots.py)
3.7 Still open - Spatial Nulls of Whole-Brain Images (brainsmash) 

 -----------------------------------------------

4. Credits

 -----------------------------------------------------------------------------------




