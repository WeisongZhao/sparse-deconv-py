[![paper](https://img.shields.io/badge/paper-nat.%20biotech.-black.svg)](https://doi.org/10.1038/s41587-021-01092-2/)
[![Github commit](https://img.shields.io/github/last-commit/WeisongZhao/sparse-deconv-py)](https://github.com/WeisongZhao/sparse-deconv-py/)
[![License](https://img.shields.io/github/license/WeisongZhao/sparse-deconv-py)](https://github.com/WeisongZhao/sparse-deconv-py/blob/master/LICENSE/)<br>
[![Twitter](https://img.shields.io/twitter/follow/weisong_zhao?label=weisong)](https://twitter.com/weisong_zhao/status/1370308101690118146)
[![GitHub watchers](https://img.shields.io/github/watchers/WeisongZhao/sparse-deconv-py?style=social)](https://github.com/WeisongZhao/sparse-deconv-py/) 
[![GitHub stars](https://img.shields.io/github/stars/WeisongZhao/sparse-deconv-py?style=social)](https://github.com/WeisongZhao/sparse-deconv-py/) 
[![GitHub forks](https://img.shields.io/github/forks/WeisongZhao/sparse-deconv-py?style=social)](https://github.com/WeisongZhao/sparse-deconv-py/)

<p>
<h2 align="center">Sparse deconvolution<sub> Python v0.3.0</sub></h2>
<!-- <h6 align="center"><sup>v1.0.3</sup></h6> -->
<!-- <h4 align="center">This repository contains the updating version of Sparse deconvolution.</h4> -->
</p>  


Official **Python** implementation of the '**Sparse deconvolution**', and the `CPU (NumPy)` and `GPU (CuPy)` calculation backend will be automatically selected. 

We haven’t tested it thoroughly, and the development is work in progress, so expect rough edges. As a result, feedback, questions, bug reports, and patches are welcome and encouraged!

It is a part of publication. For details, please refer to: "[Weisong Zhao et al. Sparse deconvolution improves the resolution of live-cell super-resolution fluorescence microscopy, Nature Biotechnology (2021)](https://doi.org/10.1038/s41587-021-01092-2)".


## Instruction

- NOTE: The MATLAB version and detailed information can be found at https://github.com/WeisongZhao/Sparse-SIM.
- NOTE: The GPU acceleration feature using CuPy requires a CUDA-based NVIDIA GPU. It could provide a ~30 times faster reconstruction speed for a `512 × 512 × 5` image stack.
- Clone/download, and run the `demo.py`

```python
from sparse_recon.sparse_deconv import sparse_deconv

im = io.imread('test.tif')
plt.imshow(im,cmap = 'gray')
plt.show()

pixelsize = 65 #(nm)
resolution = 280 #(nm)

img_recon = sparse_deconv(im, resolution / pixelsize)
plt.imshow(img_recon / img_recon.max() * 255,cmap = 'gray')
plt.show()
```

## Tested dependency 

- Python 3.7
- NumPy 1.21.4
- CuPy 9.6.0 (CUDA 11.5)
- PyWavelets 1.1.1

## Version

- v0.3.0 full Sparse deconvolution features
- v0.2.0 iterative deconvolution
- v0.1.0 initialized and started from [dzh929](https://github.com/dzh929/Sparse-SIM-python)

### Related links: [img2vid](https://github.com/WeisongZhao/img2vid/), [Adaptive filter imagej-plugin](https://github.com/WeisongZhao/AdaptiveMedian.imagej/), and [Merge channels](https://github.com/WeisongZhao/Palette.ui)



## Open source [Sparse deconvolution](https://github.com/WeisongZhao/Sparse-deconv-py)

- This software and corresponding methods can only be used for **non-commercial** use, and they are under Open Data Commons Open Database License v1.0.
- Feedback, questions, bug reports and patches are welcome and encouraged!


