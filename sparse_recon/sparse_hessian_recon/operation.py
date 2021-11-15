import  warnings
import numpy as np
try:
    import cupy as cp
except ImportError:
    cupy = None
xp = np if cp is None else cp
if xp is not cp:
    warnings.warn("could not import cupy... falling back to numpy & cpu.")

def operation_xx(gsize):
    delta_xx = xp.array([[[1, -2, 1]]], dtype = 'float32')
    xxfft = xp.fft.fftn(delta_xx, gsize) * xp.conj(xp.fft.fftn(delta_xx, gsize))
    return xxfft

def operation_xy(gsize):
    delta_xy = xp.array([[[1, -1], [-1, 1]]], dtype = 'float32')
    xyfft = xp.fft.fftn(delta_xy, gsize) * xp.conj(xp.fft.fftn(delta_xy, gsize))
    return xyfft

def operation_xz(gsize):
    delta_xz = xp.array([[[1, -1]], [[-1, 1]]], dtype = 'float32')
    xzfft = xp.fft.fftn(delta_xz,gsize) * xp.conj(xp.fft.fftn(delta_xz, gsize))
    return xzfft

def operation_yy(gsize):
    delta_yy = xp.array([[[1], [-2], [1]]], dtype = 'float32')
    yyfft = xp.fft.fftn(delta_yy,gsize) * xp.conj(xp.fft.fftn(delta_yy, gsize))
    return yyfft

def operation_yz(gsize):
    delta_yz = xp.array([[[1], [-1]], [[-1], [1]]], dtype = 'float32')
    yzfft = xp.fft.fftn(delta_yz,gsize) * xp.conj(xp.fft.fftn(delta_yz, gsize))
    return yzfft

def operation_zz(gsize):
    delta_zz = xp.array([[[1]], [[-2]], [[1]]], dtype = 'float32')
    zzfft = xp.fft.fftn(delta_zz,gsize) * xp.conj(xp.fft.fftn(delta_zz, gsize))
    return zzfft
