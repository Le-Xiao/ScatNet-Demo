from __future__ import print_function, unicode_literals,absolute_import,division

import os
import numpy as np
import json
import collections
from six.moves import range ,zip, map , reduce, filter
from .six import Path
#flag of tf backend
def is_tf_backend():
    import keras.backend as K
    return K.backend() == 'tensorflow'
    
#flag of channel last
def backend_channels_last():
    import keras.backend as K
    assert K.image_data_format() in ('channels_first','channels_last')
    return K.image_data_format() =='channels_last'
    
#'SCZYX' to 'SZYXC'
def move_channel_for_backend(X,channel):
    if backend_channels_last():
        return np.moveaxis(X,channel,-1)
    else:
        return np.moveaxis(X,channel,-1)
        
###for weights writing and reading
def load_json(fpath):
    with open(fpath,'r') as f:
        return json.load(f)
        
def save_json(data,fpath,**kwargs):
    with open(fpath,'w') as f:
        f.write(json.dumps(data,**kwargs))
        
###
def normalize(x,pmin=3,pmax=99.8,axis=None,clip=False,eps=1e-20,dtype=np.float32):
    mi=np.percentile(x,pmin,axis=axis,keepdims=True)
    ma=np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x,mi,ma,clip=clip,eps=eps,dtype=dtype)
    
def normalize_mi_ma(x,mi,ma,clip=False,eps=1e-20,dtype=np.float32):
    if dtype is not None:
        x= x.astype(dtype,copy=False)
        mi =dtype(mi)if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma =dtype(ma)if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps=dtype(eps)
    #Use numexpr to make computatation to evaluate faster    
    try:
        import numexpr
        x=numexpr.evaluate("(x-mi)/(ma-mi+eps)")
    except ImportError:
        x= (x-mi)/(ma-mi+eps)
    if clip:
        x=np.clip(x,0,1)
    return x 
    
def normalize_minmse(x,target):
    """Affine rescaling of x, such that the mean squared error to target is minimal."""
    cov= np.cov(x.flatten(),target.flatten())
    alpha= cov[0,1]/(cov[0,0]+1e-10)
    beta = target.mean() -alpha*x.mean()
    return alpha*x+beta

def _raise(e):
    raise e
    
def consume(iterator):
    collections.deque(iterator,maxlen=0)
    
def compose(*funcs):
    return lambda x:reduce(lambda f,g: g(f),funcs,x)
    
def axes_check_and_normalize(axes,length=None,disallowed=None,return_allowed=False):
    allowed='STCZYX'
    axes is not None or _raise(ValueError('axis cannot be None'))
    axes = str(axes).upper()
    #consume the final local variate
    consume(a in allowed or _raise(ValueError("invalid axis '%s',must be one of '%s'."%a)) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise("disallowed axis '%s'."%a)for a in axes)
    consume(axes.count(a)==1 or _raise(ValueError("axis '%s' occurs more than once."%a)) for a in axes)
    length is None or len(axes)==length or _raise(ValueError('axes (%s) must be of length %d.' % (axes,length)))
    return (axes,allowed) if return_allowed else axes
    
def axes_dict(axes):
    """ from axes string to dict; {dim : index}"""
    axes,allowed = axes_check_and_normalize(axes,return_allowed=True)
    return {a: None if axes.find(a)== -1 else axes.find(a) for a in allowed}
    
def move_image_axes(x,fr,to,adjust_singletons=False):
    fr = axes_check_and_normalize(fr,length=x.ndim)
    to = axes_check_and_normalize(to)
    
    fr_initial=fr 
    x_shape_initial=x.shape
    adjust_singletons= bool(adjust_singletons)
    if adjust_singletons:
        ##remove axes not present in 'to'
        #new axis
        slices = [slice(None) for _ in x.shape]
        for i,a in enumerate(fr):
            if (a not in to) and (x.shape[i]==1):
                # remove singleton axis
                slices[i] = 0
                fr = fr.replace(a,'')
        x = x[tuple(slices)]
        # add dummy axes present in 'to'
        for i,a in enumerate(to):
            if (a not in fr):
                # add singleton axis
                x = np.expand_dims(x,-1)
                fr += a
    # check the result
    if set(fr) != set(to):
        _adjusted = '(adjusted to %s and %s) ' % (x.shape, fr) if adjust_singletons else ''
        raise ValueError(
            'image with shape %s and axes %s %snot compatible with target axes %s.'
            % (x_shape_initial, fr_initial, _adjusted, to)
        )
    # to dict
    ax_from, ax_to = axes_dict(fr), axes_dict(to)
    if fr == to:
        return x
    return np.moveaxis(x, [ax_from[a] for a in fr], [ax_to[a] for a in fr])

