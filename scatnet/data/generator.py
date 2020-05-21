#make it compatible between different python versions
from __future__ import print_function, unicode_literal, absolute_import, division
from six.moves import range,zip, map, reduce, filter
from six import string_types

import numpy as np
import sys, os, warnings
# visualization of  progress bar
from tqdm import tpdm

from ..utils import _raise, consume, compose, normalize_mi_ma ,axes_dict, axes_check_and_normalize
from ..utils.six import Path
from ..io import save_training_data

from .transform import Transform, permute_axes

##Patch filter

def no_background_patches(threshold=0.4, percentile=99.9):
    """Return a patch filter to be used by :func:`create_patches` to determine for each image pair
    are eligible for sampling."""
    
    (np.isscalar(percentile) and 0 <= percentile <= 100) or (_raise(ValueError))
    (np.isscalar(threshold) and 0 <= threshold <=1) or (_raise(ValueError))
    ## locally use the API
    from scipy.ndimage.filters import maximum_filter
    #closure
    def _filter(datas, patch_size, dtype=np.float32):
        image = datas[0]
        #transform to 32bit for normalize_mi_ma
        if dtype is not None:
            image = image.astype(dtype)
        # make max filter patch_size smaller 
        # to avoid only few non-bg pixel close to image border
        patch_size=[(p//2 if p>1 else p) for p in patch_size]
        # return the  maximum of each patch
        filtered = maximum_filter(image, patch_size,mode='constant')
        # sample the pixels above the threshold (become the 0-1 mask)
        return filtered > threshold * np.percentile(image,percentile)
    return _filter
   
## Sample patches 
def sample_patches_from_multiple_stacks(datas, patch_size, n_samples, data_mask=None, patch_filter=None, verbose=False):
    """ sample matching patches of size 'patch_size' from all arrays in 'datas'"""
    
    #check the dimension
    len(patch_size)== datas[0].ndim or _raise(ValueError)
    ## self_defined error
    # check the input shape
    if not all((a.shape==datas[0].shape for a in datas)):
        raise ValueError("all input shapes must be the same: %s" % ("/".join(str(a.shape) for a in datas)))
    # check the patch_size below the border
    if not all ((0<s<=d for s,d in zip(patch_size,datas[0].shape))):
        raise ValueError("patch_size %s negative or larger than data shape %s along some dimensions" % ((str(patch_size), str(datas[0].shape))))
    # patch_filter is None, then crop the whole image; else call the func: _filter 
    if patch_filter is None:
        patch_mask = np.ones(data[0].shape, dtype=np.bool)
    else:
        patch_mask = patch_filter(datas, patch_size)
        
    if data_mask is not None:
        # TODO: Test this
        warnings.warn('Using pixel masks for raw/transformed images not tested.')
        data_mask.shape == datas[0].shape or _raise(ValueError())
        datas_mask.dtype == np.bool or _raise(ValueError())
        from scipy.ndimage.filters import minimum_filter
        patch_mask &= minimum_filter(datas_mask, patch_size, mode='constant')
        
    ## get the valid indices
    # define the border of input images to crop
    border_slices = tuple([slice(s//2,d-s+s//2+1) for s,d in zip(patch_size, datas[0].shape)])
    # apply the border to 0-1 mask, return the indexs of '1' 
    valid_inds = np.where(patch_mask[border_slices])
    
    if len(valid_inds[0]) == 0:
        raise ValueError("'patch_filter' didn't return any region to sample from")
    #get the indexs in input images; v in dimension
    valid_inds = [v + s.start for s,v in zip(border_slices, valid_inds)]
    
    #get n sample indexs, replace if n_samples is overmuch 
    sample_inds = np.random.choice(len(valid_inds[0]), n_samples, replace=len(valid_inds[0])<n_samples)
    # map back to the indexs in input images
    rand_inds= [v[sample_inds] for v in valid_inds]
    # sample every data
    res = [np.stack([data[tuple(slice(_r-(_p//2),_r+_p-(_p//2)) for _r,_p in zip(r,patch_size))] for r in zip(*rand_inds)]) for data in datas]
    
    return res
    
##create training data
#check the sample_percentile
def _valid_low_high_percentiles(ps):
    return isinstance(ps,(list,tuple,np.ndarray)) and len(ps)==2 and all(map(np.isscalar,ps)) and (0<=ps[0]<ps[1]<=100)
#check the memory for training
def _memory_check(n_required_memory_bytes, thresh_free_frac=0.5, thresh_abs_bytes=1024*1024**2):
    try:
        #ImportError
        import psutil
        # get the infos of memory
        mem =psutil.virtual_memory()
        mem_frac = n_required_memory_bytes / mem.avaiable
        if mem_frac>1:
            raise MemoryError('Not enough available memory.')
        elif mem_frac> thresh_free_frac:
            #print to error infomation
            print('Warning: will use at least %.0f MB'(%.1f%%) of available memory.\n' % (n_required_memory_bytes/1024**2,100*mem_frac), file=sys.stderr)
            sys.stderr.flush()
        except ImportError:
            if n_required_memory_bytes> thresh_abs_bytes:
                print('Warning : will use at least %.0f' MB of memory.\n' %(n_required_memory_bytes/1024**2), file=sys.stderr)
                sys.stderr.flush()
# data normalization                
def sample_percentiles(pmin=(1,3),pmax=(99.5,99.9)):
    """sample percentiles values from  a uniform distribution."""
    _valid_low_high_percentiles(pmin) or _raise(ValueError(pmin))
    _valid_low_high_percentiles(pmax) or _raise(ValueError(pmax))
    pmin[1]<pmax[0] or _raise(ValueError())
    #lambda func
    return lambda: (np.random.uniform(*pmin),np.random.uniform(*pmax))

def norm_percentiles(percentiles=sample_percentiles(), relu_last=False):
    #check if this is a callable func 
    if callable(percentiles):
        _tmp = percentiles()
        _valid_low_high_percentiles(_tmp) or _raise(ValueError(_tmp))
        get_percentiles = percentiles
    else:
        _valid_low_high_percentiles(percentiles) or _raise(ValueError(percentiles))
        get_percentiles = lambda: percentiles
    #closure    
    def _normalize(patches_x,patches_y,x,y,mask,channel):
        pmins, pmaxs = zip(*(get_percentiles()) for _ in patches_x)
        percentile_axes =None if channel is None else tuple((d for d in range(x.ndim) if d != channel))
        _perc = lambda a,p: np.percentile(a,p, axis=percentile_axes,keepdims=True)
        patches_x_norm = normalize_mi_ma(patches_x,_perc(x,pmins),_perc(x,pmaxs))
        if relu_last:
            # none below zero
            pmins =np.zeros_like(pmins)
        patches_y_norm = normalize_mi_ma(patches_y, _perc(y,pmins),_perc(y,pmaxs))
        return patches_x_norm, patches_y_norm
    
    return _normalize
# shuffle the samples
def shuffle_inplace(*arrs,**kwargs):
    seed = kwargs.pop('seed',None)
    if seed is None:
        rng = np.random
    else:
        rng = np.random.RandomState(seed=seed)
    state =rng.get_state()
    for a in arrs:
        rng.set_state(state)
        rng.shuffle(a)

# core func to create data from path   
def create_patches(
    raw_data, patch_size, n_patches_per_image, patch_axes = None,
    save_file = None, transforms= None, patch_filter = no_background_patches(),
    normalization = norm_percentiles(), shuflle = True, verbose = True,
):
    """Create normalized training data to be used for neural network training.

Parameters
----------
raw_data : :class:`RawData`
    Object that yields matching pairs of raw images.
patch_size : tuple
    Shape of the patches to be extraced from raw images.
    Must be compatible with the number of dimensions and axes of the raw images.
    As a general rule, use a power of two along all XYZT axes, or at least divisible by 8.
n_patches_per_image : int
    Number of patches to be sampled/extracted from each raw image pair (after transformations, see below).
patch_axes : str or None
    Axes of the extracted patches. If ``None``, will assume to be equal to that of transformed raw data.
save_file : str or None
    File name to save training data to disk in ``.npz`` format (see :func:`csbdeep.io.save_training_data`).
    If ``None``, data will not be saved.
transforms : list or tuple, optional
    List of :class:`Transform` objects that apply additional transformations to the raw images.
    This can be used to augment the set of raw images (e.g., by including rotations).
    Set to ``None`` to disable. Default: ``None``.
patch_filter : function, optional
    Function to determine for each image pair which patches are eligible to be extracted
    (default: :func:`no_background_patches`). Set to ``None`` to disable.
normalization : function, optional
    Function that takes arguments `(patches_x, patches_y, x, y, mask, channel)`, whose purpose is to
    normalize the patches (`patches_x`, `patches_y`) extracted from the associated raw images
    (`x`, `y`, with `mask`; see :class:`RawData`). Default: :func:`norm_percentiles`.
shuffle : bool, optional
    Randomly shuffle all extracted patches.
verbose : bool, optional
    Display overview of images, transforms, etc.

Returns
-------
tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`, str)
    Returns a tuple (`X`, `Y`, `axes`) with the normalized extracted patches from all (transformed) raw images
    and their axes.
    `X` is the array of patches extracted from source images with `Y` being the array of corresponding target patches.
    The shape of `X` and `Y` is as follows: `(n_total_patches, n_channels, ...)`.
    For single-channel images, `n_channels` will be 1.
"""
    ## images and transforms
    if transforms is None:
        transforms = []
    transforms = list(transforms)
    if patch_axes is not None:
        transforms.append(permute_axes(patch_axes))
    if len(transforms) == 0:
        transforms.append(Transform.identity())
    #get the raw pairs     
    image_pairs, n_raw_images= raw_data.generator(),raw_data.size
    # convert list of Transforms into Transform of lists
    tf= Transform(*zip(*transforms))
    # Transform : combine all transformations with raw images as input
    image_pairs =compose(*tf.generator)(image_pairs)
    
    n_tranforms= np.prod(tf.size)
    n_images = n_raw_images* n_transforms
    n_patches = n_images* n_patches_per_image
    n_required_memory_bytes = 2 * n_patches*np.prod(patch_size) * 4
    
    # memory check
    _memory_check(n_required_memory_bytes)
    
    ## summary the data
    if verbose:
        print('='*66)
        print('%5d raw images x %4d transformations   = %5d images' % (n_raw_images,n_transforms,n_images))
        print('%5d images     x %4d patches per image = %5d patches in total' % (n_images,n_patches_per_image,n_patches))
        print('='*66)
        print('Input data:')
        print(raw_data.description)
        print('='*66)
        print('Transformations:')
        for t in transforms:
            print('{t.size} x {t.name}'.format(t=t))
        print('='*66)
        print('Patch size:')
        print(" x ".join(str(p) for p in patch_size))
        print('=' * 66)
        
    sys.stdout.flush()  
    ## sample patches from each pair of transformed raw images
    # n Z Y X
    X = np.empty((n_patches,)+tuple(patch_size),dtype=np.float32)
    Y = np.empty_like(X)
    
    for i,(x,y,_axes,mask) in tqdm(enumerate(image_pairs), total=n_images,disable=(not verbose)):
        if i>= n_images:
            warnings.warn('more raw images (or transfomations there) than expected,skipping exceed images')
            break
        if i==0:
            axes = axes_check_and_normalize(_axes,len(patch_size))
            channel = axes_dict(axes)['C']
        
        axes == axes_check_and_normalize(_axes) or _raise('not all images have the same axes.')
        x.shape == y.shape or _raise(ValueError())
        mask is None or mask.shape == x.shape or _raise(ValueError())
        (channel is None or (isinstance(channel,int) and 0<=channel<x.ndim)) or _raise(ValueError())
        channel is None or patch_size[channel]==x.shape[channel] or _raise(ValueError('extracted patches must contain all channels.'))
        
        _Y,_X =sample_patches_from_multiple_stacks((y,x),patch_size,n_patches_per_image,mask,patch_filter)
        
        s=slice(i*n_patches_per_image,(i+1)*n_patches_per_image)
        
        X[s],Y[s]= normalization(_X,_Y,x,y,mask,channel)
    # shuffle
    if shuffle:
        shuffle_inplace(X,Y)
    #STCZYX    
    axes = 'SC'+axes.replace('C','')
    if channel is None:
        X = np.expand_dims(X,1)
        Y = np.expand_dims(Y,1)
    else:
        X = np.moveaxis(X,1+channel,1)
        Y = np.moveaxis(Y,1+channel,1)
        
    if save_file is not None:
        print('Saving data to %s.' % str(Path(save_file)))
        save_training_data(save_file,X,Y,axes)
            
    return X,Y,axes
    
