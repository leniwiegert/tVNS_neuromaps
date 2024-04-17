Title

 -----------------------------------------------------------------------------------

- Table of Contents -
1. Project Description
2. Installation and Set Up
	2.1 Toolbox Installation (neuromaps_first_steps.py)
	2.2 Downloads
3. User Guide
	3.1 Atlas Fetching and Simple Correlations (atlas_fetching_correlation_test_file.py)
	3.2 Data Loading, ROI Plotting and Correlation Calculation (Group Level - Whole-Brain - 
	    No Parcellation - roi_and_sc_plotting.py)
 	3.3 Randomization of the tVNS effect (rand_tVNS_vs_sham_corrected/cuneus.py?)
 	3.4 Spatial Correlations and Null Models of the rs-FC Maps of tVNS-induced Changes with PET 
 	    Receptor Maps (Group + Ind. Level - Cortical - Parcellation with Schaefer - 
 	    trans_parc_debug_3.py) --> bootstrapped?, cortical nulls distribution ("".py)
 	3.5 Comparison of Cortical and Whole-Brain Correlations (Group + Ind. Level - Parcellation
 	    with Schaefer and Tian - tian_parcellater_test.py)
  	3.6 Raincloud Plotting (raincloudplots.py)
  	3.7 Spatial Nulls of Whole-Brain Images (brainsmash) 
  4. Credits
  
 -----------------------------------------------------------------------------------
 
- Project's Title: Code Overview -

1. Project Description: What does my code do, why did I use certain technologies, challenges and 	
    improvements
    	- Motivation
	- Why did I build this project?
	- What problem does it solve?
	- What did I learn?
	- What makes my project stand out?
  
 -----------------------------------------------

2. Installation and Set Up:
2.1 Toolbox Installation (neuromaps_first_steps.py)
- Download connectome workbench: https://www.humanconnectome.org/software/get-connectome-workbench)
- Create environment and download neuromaps by following the installation steps in the documentation: 	
  Link
2.2 Downloads
- Download the following data files from the NAS:
	- rs-FC maps of tVNS-induced changes (4D_rs_fCONF_del_taVNS_sham.nii)
	- Group level image with GM mask (combined_mask.nii.gz)
	- Individual level images without GM mask (volume_1.nii to volume_41.nii)
	
 -----------------------------------------------
	
3. User Guide:
3.1 Atlas Fetching and Simple Correlations (atlas_fetching_correlation_test_file.py)
	- Check out currently available files of Neuromaps 
	- Example for D2 receptor map
3.2 Data Loading, ROI Plotting and Correlation Calculation (roi_and_sc_plotting.py)
	- Load tVNS data and check out its properties (header, affine, ...)
	- Create and load seperate files for single subject data (individual data of each participant, 
	  named by their ID)
	- Create mean image of individual data files (necessary?)
	- Try different plotting options 
	- Calculate spatial correlations of the rs-FC maps of tVNS-induced changes with 11 PET receptor maps 
	  on group and individual level
	- Run on Cuneus and doublecheck plotting part 
 3.3 Randomization of the tVNS effect (rand_tVNS_vs_sham_corrected.py)
 3.4 Spatial Correlations and Null Models of the rs-FC Maps of tVNS-induced Changes with PET Receptor Maps 
 (Group + Ind. Level - Cortical - Parcellation with Schaefer - trans_parc_debug_3.py)
	 - Load and prep data as previously shown 
	 - Transform data and annotation from MNI152 (volumetric) to fsLR (surface-based) space
	- Parcellate data and annotation with Schaefer2018 (Cortical Brain Atlas) 
	- Group Level: Calculate nulls and spatial correlations for the rs-FC maps of tVNS-induced changes with 
	  11 PET receptor maps 
	- Individual Level: Same as on group level
	- Heatmap plotting on group and individual level
3.5 Comparison of Cortical and Whole-Brain Correlations 
(Group + Ind. Level - Parcellation with Schaefer and Tian - tian_parcellater_test.py)
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

TO DO:
- Adjust each code to Cuneus/make Cuneus version
- Check notes in readme
- Add description and title
- Tian auf cuneus runterladen

 -----------------------------------------------------------------------------------






