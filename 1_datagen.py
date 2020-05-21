from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt

import argparse
from tifffile import imread
from scatnet.utils import download_and_extract_zip_file, plot_some
from scatnet.data import RawData, create_patches



## Generate training data 


raw_data = RawData.from_folder (
    basepath    = 'G:/ScatterNet/scattering_data/3D_droplet_bayesian/0519/registration/3/bead_train/train/refine_register/1/refine/input/train',
    source_dirs = ['LR'],
    target_dir  = 'HR',
    axes        = 'ZYX',
)



X, Y, XY_axes = create_patches (
    raw_data            = raw_data,
    patch_size          = (16,64,64),
    n_patches_per_image = 600,
    save_file           = 'D:/NN/CSBDeep-CE/data/bead_200327/my_training_data.npz',
)


assert X.shape == Y.shape
print("shape of X,Y =", X.shape)
print("axes  of X,Y =", XY_axes)

## Show

for i in range(10):
    plt.figure(figsize=(16,4))
    sl = slice(8*i, 8*(i+1)), 0
    plot_some(X[sl],Y[sl],title_list=[np.arange(sl[0].start,sl[0].stop)])
    plt.show()
None;

