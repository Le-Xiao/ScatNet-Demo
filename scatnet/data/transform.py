#make it compatible between different python versions
from __future__ import print_function, unicode_literals, absolute_import,division
from six.moves import range,zip,map, reduce, filter
from six import string_types

import numpy as np
from collections import namedtuple
import sys,os,warnings

from ..utils import _raise,consume,axes_check_and_normalize,axes_dict,move_image_axes

class Transform(namedtuple('Transform',('name','generator','size'))):
    """Extension of :func:`collections.namedtuple` with three fields: `name`, `generator`, and `size`."""
    
    @staticmethod
    def identity():
        """
        Returns
        -------
        Transform
            Identity transformation that passes every input through unchanged.
        """
        #closure
        def _gen(inputs):
            for d in inputs:
                yield d
        return Transform('Identity', _gen, 1)
        
def permute_axes(axes):
    """Transformation to permute images axes.
    axes:str
        target axes, to which the input images will be permuted.
        
    Returns
    -------
    Transform
        Returns a : class:'Transform' object whose 'generator' will perform the axes permutation of 'x','y' and 'mask'
    """
    axes = axes_check_and_normalize(axes)
    def _generator(inputs):
        for x,y,axes_in,mask in inputs:
            axes_in =axes_check_and_normalize(axes_in)
            if axes_in !=axes:
                x=move_image_axes(x,axes_in,axes,True)
                y=move_image_axes(y,axes_in,axes,True)
                if mask is not None:
                    mask =move_image_axes(mask,axes_in,axes)
            yield x,y,axes,mask
            
    return Transform('Permute axes to %s' %axes,_generator,1)
    
def crop_images(slices):
    """Transformation to crop all images (and mask).

    Note that slices must be compatible with the image size.

    Parameters
    ----------
    slices : list or tuple of slice
        List of slices to apply to each dimension of the image.

    Returns
    -------
    Transform
        Returns a :class:`Transform` object whose `generator` will
        perform image cropping of `x`, `y`, and `mask`.

    """
    slices = tuple(slices)
    def _generator(inputs):
        for x,y,axes,mask in inputs:
            axes =axes_check_and_normalize(axes)
            len(axes)== len(slices) or _raise(ValueError())
            yield x[slices],y[slices],axes,(mask[slices] if mask is not None else None)
            
    return Transform('Crop images(%s)' % str(slices),_generator,1)
    
