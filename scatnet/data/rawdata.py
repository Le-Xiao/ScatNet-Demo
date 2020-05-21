#make it compatible between different python versions
from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import zip

from tiffile import imread
# to build a named tuple
from collections import namedtuple
# chain the iteritems
from itertools import chain

from ..utils import _raise, consume, axes_check_and_normalize
from ..utils.six import Path, FileNotFoundError

class RawData(namedtuple('RawData',('generator','size','description'))):
    """ 'collections.namedtuple' with three fields: 'generator','size', and 'description'"""
    
    @staticmethod
    def from_folder(basepath,source_dirs,target_dir,axes='CZYX',pattern='*.tif*'): #tif and tiff
        """get pairs of corresponding TIFF images from floders.
        Two images correspond to each other if they have the same file name, but are located in different folders.

        Parameters
        ----------
        basepath : str
            Base folder that contains sub-folders with images.
        source_dirs : list or tuple
            List of folder names relative to `basepath` that contain the source images (e.g., with low SNR).
        target_dir : str
            Folder name relative to `basepath` that contains the target images (e.g., with high SNR).
        axes : str
            Semantics of axes of loaded images (assumed to be the same for all images).
        pattern : str
            Glob-style pattern to match the desired TIFF images.
        """
        p= Path(basepath)
        #get the paired existing raw images
        pairs = [(f, p/target_dir/f.name) for f in chain(*((p/source_dir).glob(pattern) for source_dir in source_dirs))]
        #
        len(pairs)>0 or _raise(FileNotFoundError("Didn't find any images."))
        consume(t.exists() or _raise(FileNotFoundError(t)) for s,t in pairs)
        n_images = len(pairs)
        description ="{p}:target='{o}',sources={s},axes='{a}',pattern='{pt}'".format(p=basepath,s=list(source_dirs),o=target_dir,a=axes,pt=pattern)
        #closure
        def _gen():
            for fx,fy in pairs:
                x,y =imread(str(fx)),imread(str(fy))
                len(axes) >= x.dim or _raise(ValueError())
                yield x,y,axes[-x.ndim:],None
                
        return RawData(_gen,n_images,description)
        