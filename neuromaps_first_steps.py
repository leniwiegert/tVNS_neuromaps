# Author: Lena Wiegert

# --------------------- NEUROMAPS INSTALLATION --------------------- #

# This package requires Python 3.7+

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
# Restart shell.
# Check with:
echo $PATH
# New path should be added.
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





