from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread
from scatnet.utils import axes_dict, plot_some, plot_history
from scatnet.utils.tf import limit_gpu_memory
from scatnet.io import load_training_data
from scatnet.models import Config, SCAT



# # Training data
(X,Y), (X_val,Y_val), axes = load_training_data('D:/NN/CSBDeep-CE/data/drosophila_200328/selected/my_training_data.npz', validation_split=0.15, verbose=True)

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

##set config
config = Config(axes, n_channel_in, n_channel_out,train_steps_per_epoch=300)

print(config)


model = SCAT(config, 'drosophila_200328_depth4', basedir='models')

# # Training
# You can start TensorBoard from the current working directory with `tensorboard --logdir=.`
# Then connect to [http://localhost:6006/](http://localhost:6006/) with your browser.

history = model.train(X,Y, validation_data=(X_val,Y_val))


# Plot final training history (available in TensorBoard during training):

print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);

