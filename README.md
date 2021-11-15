

<p>
<h2 align="center">Sparse deconvolution<sub> Python v0.3.0</sub></h2>
<!-- <h6 align="center"><sup>v1.0.3</sup></h6> -->
<!-- <h4 align="center">This repository contains the updating version of Sparse deconvolution.</h4> -->
</p>  





Official **Python** implementation of the '**Sparse deconvolution**', and the `CPU (NumPy)` and `GPU (CuPy)` calculation backend will be automatically selected. 

We haven’t tested it throughout, and the development is work in progress, so expect rough edges. As a result, feedback, questions, bug reports, and patches are welcome and encouraged!

It is a part of publication. For details, please refer to: "[Weisong Zhao et al. Sparse deconvolution improves the resolution of live-cell super-resolution fluorescence microscopy, Nature Biotechnology (2021)](https://doi.org/10.1038/s41587-021-01092-2)".

## Instruction
- The MATLAB version and detailed information can be found at https://github.com/WeisongZhao/Sparse-SIM
-  The GPU acceleration requires a CUDA-compatible NVIDIA GPU.
- Clone/download, and run the `demo.py`
- demo:
```python
from sparse_recon.sparse_deconv import sparse_deconv

im = io.imread('test.tif')
plt.imshow(im,cmap ='gray')
plt.show()

pixelsize = 65 #(nm)
resolution = 280 #(nm)

img_recon = sparse_deconv(im, resolution/pixelsize)
plt.imshow(img_recon / img_recon.max() * 255,cmap = 'gray')
plt.show()
```



## Depencency 

- Python
- NumPy
- CuPy
- PyWavelets

## Version

- v0.3.0 full Sparse deconvolution features
- v0.2.0 iterative deconvolution
- v0.1.0 initialized and started from [dzh929](https://github.com/dzh929/Sparse-SIM-python)

### Related links: [img2vid](https://github.com/WeisongZhao/img2vid/), [Adaptive filter imagej-plugin](https://github.com/WeisongZhao/AdaptiveMedian.imagej/), and [Merge channels](https://github.com/WeisongZhao/Palette.ui)



## Open source [Sparse deconvolution](https://github.com/WeisongZhao/Sparse-deconv-py)

- This software and corresponding methods can only be used for **non-commercial** use, and they are under Open Data Commons Open Database License v1.0.
- Feedback, questions, bug reports and patches are welcome and encouraged!


