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


import eigenstrapping as eigen
from eigen import data
from eigen.datasets import load_surface_examples


