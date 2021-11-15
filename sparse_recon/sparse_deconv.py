import cupy as cp
import numpy as np
import warnings
import time

from matplotlib import pyplot as plt

from sparse_recon.sparse_hessian_recon.sparse_hessian_recon import sparse_hessian
from sparse_recon.iterative_deconv.iterative_deconv import iterative_deconv
from sparse_recon.iterative_deconv.kernel import Gauss
from utils.background_estimation import background_estimation
from utils.upsample import spatial_upsample, fourier_upsample
try:
    import cupy as cp
except ImportError:
    cupy = None
xp = np if cp is None else cp
if xp is not cp:
    warnings.warn("could not import cupy... falling back to numpy & cpu.")
def sparse_deconv(im, sigma, sparse_iter = 100, fidelity = 150, sparsity = 10, tcontinuity = 0.5,
                          background = 1, deconv_iter = 7, deconv_type = 1,
                          up_sample = 0,):

    """Sparse deconvolution.
   	----------
   	It is an universal post-processing framework for 
   	fluorescent (or intensity-based) image restoration, 
   	including xy (2D), xy-t (2D along t axis), 
   	and xy-z (3D) images. 
   	It is based on the natural priori 
   	knowledge of forward fluorescent 
   	imaging model: sparsity and 
   	continuity along xy-t(z) axes.
   	----------
    Parameters
    ----------
    im : ndarray
       Input image (can be N dimensional).
    sigma : 1/2/3 element(s) list
       The point spread function size in pixel.
    sparse_iter:  int, optional
         the iteration of sparse hessian {default: 100}
    fidelity : int, optional
       fidelity {default: 150}
    tcontinuity  : optional
       continuity along z-axial {default: 0.5}
    sparsity :  int, optional
        sparsity {default: 10}
    background:int, optional
        background estimation {default:1}:
        when background is none, 0
        when background is Weak background (HI), 1
        when background is Strong background (HI), 2
        when background is Weak background (LI), 3
        when background is with background (LI), 4
        when background is Strong background (LI), 5
    deconv_iter : int, optional
        the iteration of deconvolution {example:7}
    deconv_type : int, optional
       choose the different type deconvolution:
       0: No deconvolution
       1: LandWeber deconxolution
       2: Richardson-Lucy deconvolution

    Returns
    -------
    img_last : ndarray
       The sparse deconvolved image.

    Examples
    --------
    >>> from sparse_recon.sparse_deconv import sparse_deconv
	>>> from skimage import io
    >>> im = io.imread('test.tif')
	>>> img_recon = sparse_deconv(im, [5,5])
    References
    ----------
      [1] Weisong Zhao et al. Sparse deconvolution improves
      the resolution of live-cell super-resolution 
      fluorescence microscopy, Nature Biotechnology (2021),
      https://doi.org/10.1038/s41587-021-01092-2
    """
    if not sigma:
        print("The PSF's sigma is not given, turning off the iterative deconv...")
        deconv_type = 0
    im = np.array(im, dtype = 'float32')
    im = im / (im.max())
    index = im.max()
    if background == 2:
        backgrounds = background_estimation(im / 2)
        im=im- backgrounds
    elif background == 1:
        backgrounds = background_estimation(im / 2.5)
        im=im- backgrounds
    elif background== 4:
        medVal = np.mean(im)
        im[im> medVal] = medVal
        backgrounds = background_estimation(im)
        im = im - backgrounds
    elif background == 5:
        medVal = np.mean(im)/2
        im[im > medVal] = medVal
        backgrounds = background_estimation(im)
        im = im - backgrounds
    elif background == 3:
        medVal = np.mean(im) / 2.5
        im[im > medVal] = medVal
        backgrounds = background_estimation(im)
        im = im - backgrounds

    im = im / (im.max())
    # plt.imshow(im*255,cmap ='gray')
    # plt.show()
    im[im < 0] = 0

    if up_sample == 1:
        im = fourier_upsample(im)
    elif up_sample == 2:
        im = spatial_upsample(im)
    im = im / (im.max())
    start = time.clock()
    img_sparse = sparse_hessian(im, sparse_iter, fidelity, sparsity, tcontinuity)
    end = time.clock()
    print('sparse hessian time')
    print(end - start)
    img_sparse = img_sparse / (img_sparse.max())
    if deconv_type == 0:
        img_last = img_sparse
        return index * img_last
    else:
        start = time.clock()
        kernel = Gauss(sigma)
        img_last = iterative_deconv(img_sparse, kernel, deconv_iter, rule = deconv_type)
        end = time.clock()
        print('deconv time')
        print(end - start)
        return index * img_last

























