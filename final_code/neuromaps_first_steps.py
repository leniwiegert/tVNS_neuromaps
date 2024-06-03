# Author: Lena Wiegert
# This code is part of the Master Thesis "Placing the Effects of tVNS into Neurobiological Context"

# --------------------- NEUROMAPS INSTALLATION --------------------- #

# This package requires Python 3.7+
# Add my version + extra environment

# --- Connectome Workbench ---
# Download connectome workbench: https://www.humanconnectome.org/software/get-connectome-workbench

# -- Windows/Linux:
# Follow the steps in README.txt

# -- For macOS --
# Check shell with:
printenv SHELL
# Change to bash as default if needed:
chsh -s /bin/bash
# Set environment variable (make sure to add your own path):
echo 'export PATH=$PATH:/Users/leni/Documents/Master/python/workbench/bin_macosx64' >> ~/.bash_profile
# Restart shell or run 'source ~/.bash_profile'
# Check with:
echo $PATH
# New path should be added.
# If not try and enter your password:
# echo 'export PATH=$PATH:/Users/leni/Documents/Master/python/workbench/bin_macosx64' | sudo tee -a ~/.bash_profile
# Make sure Connectome Workbench is properly installed:
wb_command -version

# --- Neuromaps Basic Installation ---

git clone https://github.com/netneurolab/neuromaps.git
cd neuromaps
pip install .


'''
# Add new code to GitHub:
# Clone repository:
git clone https://github.com/your-username/your-repository.git
# Move into directory of your cloned repository:
cd your-repository
# Commit your changes with a message:
git commit -m "New commit"
# Push changes to GitHub:
git push origin master
'''

#--------------------- FETCHING ATLASES AND ANNOTATIONS ---------------------  #

# example for fetching atlases from neuromaps:
from neuromaps import datasets
# general function: neuromaps.datasets.fetch_atlas()
fslr = datasets.fetch_atlas(atlas='fslr', density='32k')


