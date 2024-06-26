
Placing the Effects of Transcutaneous Vagus Nerve Stimulation into Neurobiological Context 

 -----------------------------------------------------------------------------------

- Table of Contents -

1. Project Description
2. Installation and Set Up
	2.1 Toolbox Installation (neuromaps_first_steps.py)
	2.2 Downloads
3. User Guide
	3.1 Atlas Fetching and Simple Correlations (atlas_fetching.py)
	3.2 Data Loading, ROI Plotting and Correlation Calculation (data_prep_final.py)
 	3.3 Randomization of the tVNS effect (permutation_tVNS_effect_final.py)
	3.4  Spatial Null Models of the Maps of tVNS-induced Changes with PET
 	    Receptor Maps - Group Level (cortical_nulls_final.py)
 	3.5 Spatial Null Models of the Maps of tVNS-induced Changes with PET
 	    Receptor Maps - Individual Level (sc_groupind_parc_cortical_final.py)
	3.6 ROI-based comparison of Cortical and Subcortical Datasets + Subnetwork Analysis 
	    (subnetworks_parc_cortical.py)
 	3.7 Comparison of Cortical and Subcortical Correlations 
	    (sc_groupind_parc_cort_subcort.py)
  
 -----------------------------------------------------------------------------------
 
- Code Overview -

1. Project Description:

This project explores the neurobiological mechanisms underlying the effects of non-invasive transcutaneous Vagus Nerve 
Stimulation (tVNS). Using fMRI data and advanced analytical techniques, the study aims to uncover associations between 
tVNS-induced brain responses and neurobiological factors, such as receptor density patterns. The open-access toolbox 
Neuromaps for accessing, transforming, and analyzing structural brain maps (Markello et al., 2022) is employed and 
neurobiological receptor distribution data from in-vivo Positron Emission Tomography (PET) maps from the Neuromaps 
database are used. By examining individual patterns of brain response to tVNS compared to group-level data, we aim to 
elucidate heterogeneity in tVNS responses and advance the understanding of tVNS as a potential therapeutic intervention. 
The project code, written in Python, statistically compares brain maps of tVNS-induced changes to PET receptor density 
maps at both individual and group levels.

 -----------------------------------------------

2. Installation and Set Up:

2.1 Toolbox Installation (neuromaps_first_steps.py)
- Download connectome workbench: 
  https://www.humanconnectome.org/software/get-connectome-workbench)
- Create environment and download neuromaps by following the installation steps in the documentation: 	
  https://neuromaps.readthedocs.io/en/latest/installation.html

2.2 Downloads
- Download the following data files from the HippocampNAS:
  Data files 1 and 2 can also be found in their original NAS folders, but for simplification they were
  uploaded in this project's folder: /TUE_general/Projects/Thesis_work/LenaWiegert/Data_LW
	1) Maps of tVNS-induced changes (4D_rs_fCONF_del_taVNS_sham.nii) - collected by Teckentrup et al. (2021)
	2) Gray matter mask (out_GM_p_0_15.nii) - from our group 
	3) Group level image with GM mask (combined_mask.nii.gz) 
	4) Individual datasets without GM mask (volume_1.nii to volume_41.nii) 
	
 -----------------------------------------------
	
3. User Guide:

3.1 Atlas Fetching and Simple Correlations (atlas_fetching.py)
	- Overview on how to fetch brain atlases (annotations, brain maps) from neuromaps
	- First try of spatial null models to test the statistical correlation between two
	  brain maps (whole-brain level) based on the Alexander-Bloch Null Model

3.2 Data Loading, ROI Plotting and Correlation Calculation (data_prep_final.py)
	- Load tVNS data and check out its properties (header, affine, ...)
	- Create and load seperate files for individual datasets (for each participant, named by their ID))
	- Create mean image of individual data files
	- Try different plotting options 
	- Calculate spatial correlations of the maps of tVNS-induced changes with 11 PET receptor maps 
	  on group and individual level 

 3.3 Randomization of the tVNS effect (permutation_tVNS_effect_final.py) (A)
	- Create a permutation design that preserves spatial data structure 
	- Randomize the algebraic sign of individual maps of tVNS-induced changes (whole-brain maps, group level)
	- Calculate the spatial correlations of the randomized maps with the receptor maps, permute 1000 times
	- Plot and compare to the non-randomized (original) correlation value

 3.4 Spatial Null Models of the Maps of tVNS-induced Changes with PET Receptor Maps - Group Level
     (cortical_nulls_final.py) (B)
	- Load and prep data as previously shown 
	- Transform whole-brain data and receptor maps (annotations) from MNI152 (volumetric) to fsLR (surface-based) space
	- Parcellate data and annotation with Schaefer2018 (Human Cerebral Cortex Atlas) 
	- Calculate nulls and spatial correlations for the maps of tVNS-induced changes with PET receptor maps 
	  for noradrenaline, dopamine and serotonin

 3.5 Spatial Null Models of the Maps of tVNS-induced Changes with PET Receptor Maps - Individual Level
    (sc_groupind_parc_cortical_final.py) (C) 
	- Load and prep data as previously shown 
	- Transform whole-brain data and receptor maps (annotations) from MNI152 (volumetric) to fsLR (surface-based) space
	- Parcellate data and annotation with Schaefer2018 (Human Cerebral Cortex Atlas) 
	- Group Level: Calculate nulls and spatial correlations for the maps of tVNS-induced changes with 
	  PET receptor maps for noradrenaline, dopamine and serotonin (as shown in the previous file)
	- Individual Level: Same as on group level
	- Heatmap plotting on individual level for each receptor group

3.6 Subnetwork Analysis (subnetworks_parc_cortical.py) (D)
	- Load and prep data as previously shown 
	- Parcellation on cortical and subcortical level
	- ROI-based comparison of the spatial correlations of parcellated cortical maps of tVNS-induced changes
	  to the same data as parcellated subcortical maps on ROI-basis (group level)
	- Analysis based on the 7 subnetworks included in the Schaefer Atlas (Yeo 2011)

3.7 Comparison of Cortical and Subcortical Results (sc_groupind_parc_cort_subcort.py) (E)
	- Download Melbourne Subcortical Brain Atlas (Ye Tian): 
	https://www.nitrc.org/frs/download.php/13364/Tian2020MSA_v1.4.zip
	- Define your atlas_directory (line 221)
	- Load and prep data as previously shown 
	- Cortical data (group + individual):
		- Transformation and parcellation of cortical images with Schaefer2018
		- Calculation of the spatial correlations with 11 PET receptor maps 
	- Subcortical images (group + individual):
		- Parcellation of subcortical images with Tian2020 (Melbourne Subcortical Brain Atlas) 
		- Calculation of the spatial correlations with 11 PET receptor maps 
	- Plotting for comparison of cortical and subcortical data on group and individual level


 -----------------------------------------------------------------------------------

4. Credits:

This code was written by Lena Wiegert under supervision of Corinna Schulz and Dr. Anne Kühnel
within the neuroMADLAB (Neuroscience of Motivation, Action, and Desire Laboratory, Department 
of Psychiatry & Psychotherapy, University of Bonn & Tübingen). 


