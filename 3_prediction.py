from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt


from tifffile import imread
from scatnet.utils import Path, download_and_extract_zip_file, plot_some
from scatnet.io import save_tiff_imagej_compatible
from scatnet.models import SCAT
#read raw image
x = imread('G:/ScatterNet/scattering_data/3D_droplet_bayesian/0519/registration/3/bead_train/train/refine_register/1/refine/input/validation/test/x_z2step/Reslice of beads_200328_selected_result_last.tif')


axes = 'ZYX'



##  get trained model
model = SCAT(config=None, name='beads_200328_selected', basedir='models')


## Apply SCAT network to raw image

# Predict the restored image (image will be successively split into smaller tiles if there are memory issues).

restored = model.predict(x, axes)


## Save restored image
Path('results').mkdir(exist_ok=True)
save_tiff_imagej_compatible('results/%s_result_reslice—last_step_xz.tif' % model.name, restored, axes)

