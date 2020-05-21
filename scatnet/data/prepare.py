##
from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map , reduce, filter

from ..utils import _raise, consume, normalize_mi_ma, axes_dict,axes_check_and_normalize, move_image_axes
import warnings
import numpy as np

from six import add_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty

@add_metaclass(ABCMeta)
class Normalizer():
    """Abstract base class for normalization"""
    @abstractmethod
    def before(self,x,axes):
        """Normalization of the raw input image (method stub).

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Raw input image.
        axes : str
            Axes of input image x

        Returns
        -------
        :class:`numpy.ndarray`
            Normalized input image with suitable values for neural network input.
        """
    @abstractmethod
    def after(self,mean,scale,axes):
        """Possible adjustment of predicted restored image (method stub).

        Parameters
        ----------
        mean : :class:`numpy.ndarray`
            Predicted restored image or per-pixel ``mean`` of Laplace distributions
            for probabilistic model.
        scale: :class:`numpy.ndarray` or None
            Per-pixel ``scale`` of Laplace distributions for probabilistic model (``None`` otherwise.)
        axes : str
            Axes of ``mean`` and ``scale``

        Returns
        -------
        :class:`numpy.ndarray`
            Adjusted restored image(s).
        """

    @abstractproperty
    def do_after(self):
        """bool : Flag to indicate whether :func:`after` should be called."""
        
class NoNormalizer(Normalizer):
    """not normalize"""
    def __init__(self,do_after=False):
        #protected
        self._do_after = do_after
        
    def before(self,x,axes):
        retrun x
        
    def after(self,mean,scale ,axes):
        self.do_after or _raise(ValueError())
        return mean, scale
    
    @property
    def do_after(self):
        return self._do_after

class PercentileNormalizer(Normalizer):
        """Percentile-based image normalization.
        Parameters
        ----------
        pmin : float
            Low percentile.
        pmax : float
            High percentile.
        do_after : bool
            Flag to indicate whether to undo normalization (original data type will not be restored).
        dtype : type
            Data type after normalization.
        kwargs : dict
            Keyword arguments for :func:`csbdeep.utils.normalize_mi_ma`.
        """
        
        def __init__(self,pmin=2,pmax=99.8,do_after=True,dtype=np.float32,**kwargs):
            (np.isscalar(pmin) and np.isscalar(pmax) and 0<=pmin <pmax<=100) or _raise(ValueError())
            self.pmin=pmin
            self.pmax=pmax
            self._do_after=do_after
            self.dtype = dtype
            self.kwargs=kwargs
            
        def before(self,x,axes):
            self.axes_before = axes_check_and_normalize(axes,x.ndim)
            #get the index except 'C',for normalization
            axis= tuple(d for d,a in enumerate(self,axes_before) if a !='C')
            #get the mi/ma pixel values
            #the axes which are reduced are left in the result as dimensions with size one
            self.mi = np.percentile(x,self.pmin,axis=axis,keepdims=True).astype(self.dtype,copy=False)
            self.ma = np.percentile(x,self.pmax,axis=axis,keepdims=True).astype(self.dtype,copy=False)
            return normalize_mi_ma(x,self.mi,self.ma,dtype=self.dtype,**self.kwargs)
            
        def after(self,mean,scale,axes):
            """Undo percentile-based normalization to map restored image to similar range as imput image."""
            self.do_after or _raise(ValueError())
            self.axes_after =axes_check_and_normalize(axes,mean.ndim)
            mi= move_image_axes(self.mi,self.axes_before,self.axes_after,True)
            ma= move_image_axes(self.ma,self.axes_before,self.axes_after,True)
            alpha = ma-mi
            beta= mi
            return ((alpha*mean+beta).astype(self.dtype,copy=False),(alpha*scale).astype(self.dtype,copy=False) if scale is not None else None)
            
        @property
        def do_after(self):
            return self._do_after
            
@add_metaclass(ABCMeta)
class Resizer():
    """Abstract base class for resizing"""
    
    @abstractmethod
    def before(self,x,axes,axes_div_by):
        """resize the raw input images
        axes_div_by : iterable of int
            Resized image must be evenly divisible by the provided values for each axis.
        """
        
    @abstractmethod
    def after(self,x,axes):
        """resize the restored image"""
        
class NoResizer(Resizer):
    def before(self,x,axes,axes_div_by):
        axes=axes_check_and_normalize(axes,x.ndim)
        consume(
        (s%div_n==0) or _raise(ValueError('%d (axis %s) is not divisible by %d.' %(s,a,div_n)))
        for a,div_n,s in zip(axes,axes_div_by,x.shape)
        )
        return x
        
    def after(self,x,axes):
        return x
        
class PadAndCropResizer(Resizer):
    """resize image by padding and cropping"""
    def __init__(self,mode='reflect',**kwargs):
        self.mode =mode
        self.kwargs=kwargs
        
    def before(self,x,axes,axes_div_by):
        """Pad input image to divide evenly"""
        axes=axes_check_and_normalize(axes,x.ndim)
        #protected func
        def _split(v):
            a=v//2
            #forward padding v/2,backward padding v/2
            return a,v-a
        self.pad={
            #specialize in all dim
            a: _split((div_n-s%div_n)%div_n)
            for a,div_n,s in zip(axes,axes_div_by,x.shape)
        }
        x_pad =np.pad(x,tuple(self.pad[a] for a in axes), mode=self.mode,**self.kwargs)
        return x_pad
    
    def after(self,x,axes):
        """ crop restored image to retain size of input image."""
        axes =axes_check_and_normalize(axes,x.ndim)
        all(a in self.pad for a in axes) or _raise(ValueError())
        #get the crop index
        # reset the padding
        crop = tuple(
            slice(p[0],-p[1] if p[1]>0 else None)
            for p in (self.pad[a] for a in axes)
        )
        return x[crop]
