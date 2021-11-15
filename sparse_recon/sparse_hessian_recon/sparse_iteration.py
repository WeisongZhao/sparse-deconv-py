import  warnings
import numpy as np
try:
    import cupy as cp
except ImportError:
    cupy = None
xp = np if cp is None else cp
if xp is not cp:
    warnings.warn("could not import cupy... falling back to numpy & cpu.")

def forward_diff(data, step, dim):
    # data --- input image(gpu array!!!)
    # step
    # dim --- determine which is the dimension to calculate derivate
    # dim = 0 --> z axis
    # dim = 1 --> y axis
    # dim = 2 --> x axis

    assert dim <= 2
    r, n, m = np.shape(data)
    size = np.array((r, n, m))
    position = xp.zeros(3, dtype = 'float32')
    temp1 = xp.zeros(size + 1, dtype = 'float32')
    temp2 = xp.zeros(size + 1, dtype = 'float32')

    size[dim] = size[dim] + 1
    position[dim] = position[dim] + 1

    temp1[position[0]:size[0], position[1]:size[1], position[2]:size[2]] = data
    temp2[position[0]:size[0], position[1]:size[1], position[2]:size[2]] = data

    size[dim] = size[dim] - 1
    temp2[0:size[0], 0:size[1], 0:size[2]] = data
    temp1 = (temp1 - temp2) / step
    size[dim] = size[dim] + 1

    out = temp1[position[0]:size[0], position[1]:size[1], position[2]:size[2]]
    return -out


def back_diff(data, step, dim):
    # data --- input image(gpu array!!!)
    # step
    # dim --- determine which is the dimension to calculate derivate
    # dim = 0 --> z axis
    # dim = 1 --> y axis
    # dim = 2 --> x axis
    assert dim <= 2
    r, n, m = np.shape(data)
    size = np.array((r, n, m))
    position = np.zeros(3,dtype='float32')
    temp1 = xp.zeros(size + 1,dtype='float32')
    temp2 = xp.zeros(size + 1,dtype='float32')

    temp1[position[0]:size[0], position[1]:size[1], position[2]:size[2]] = data
    temp2[position[0]:size[0], position[1]:size[1], position[2]:size[2]] = data

    size[dim] = size[dim] + 1
    position[dim] = position[dim] + 1

    temp2[position[0]:size[0], position[1]:size[1], position[2]:size[2]] = data
    temp1 = (temp1 - temp2) / step
    size[dim] = size[dim] - 1
    out = temp1[0:size[0], 0:size[1], 0:size[2]]
    return out

def shrink(x, L):
    s = xp.abs(x)
    xs = xp.sign(x) * xp.maximum(s - 1 / L, 0) 
    return xs

def iter_xx(g, bxx, para, mu):
    """

    :param g: input image
    :param bxx: use to update dxx
    :param para: regularization parameters
    :param mu: parameter which is used to tune fidelity term
    :return: Lxx,bxx
    """
    gxx = back_diff(forward_diff(g, 1, 1), 1, 1)
    dxx = shrink(gxx + bxx, mu)
    bxx = bxx + (gxx - dxx)
    Lxx = para * back_diff(forward_diff(dxx - bxx, 1, 1), 1, 1)
    return Lxx, bxx


def iter_xy(g, bxy, para, mu):
    """

    :param g: input image
    :param bxy: use to update dxy
    :param para: regularization parameters
    :param mu: parameter which is used to tune fidelity term
    :return: Lxy,bxy
    """
    gxy = forward_diff(forward_diff(g, 1, 1), 1, 2)
    dxy = shrink(gxy + bxy, mu)
    bxy = bxy + (gxy - dxy)
    Lxy = para * back_diff(back_diff(dxy - bxy, 1, 2), 1, 1)
    return Lxy, bxy


def iter_xz(g,bxz,para,mu):
    """

    :param g: input image
    :param bxz: use to update dxz
    :param para: regularization parameters
    :param mu: parameter which is used to tune fidelity term
    :return: Lxz,bxz
    """
    gxz = forward_diff(forward_diff(g, 1, 1), 1, 0)
    dxz = shrink(gxz + bxz, mu)
    bxz = bxz + (gxz - dxz)
    Lxz = para * back_diff(back_diff(dxz - bxz, 1, 0), 1, 1)
    return Lxz, bxz

def iter_yy(g, byy, para, mu):
    """

    :param g: input image
    :param byy: use to update dxy
    :param para: regularization parameters
    :param mu: parameter which is used to tune fidelity term
    :return: Lyy,byy
    """
    gyy = back_diff(forward_diff(g, 1, 2), 1, 2)
    dyy = shrink(gyy + byy, mu)
    byy = byy + (gyy - dyy)
    Lyy = para * back_diff(forward_diff(dyy - byy, 1, 2), 1, 2)
    return Lyy, byy

def iter_yz(g,byz,para,mu):
    """

    :param g: input image
    :param byz: use to update dxy
    :param para: regularization parameters
    :param mu: parameter which is used to tune fidelity term
    :return: Lyz,byz
    """
    gyz = forward_diff(forward_diff(g, 1, 2), 1, 0)
    dyz = shrink(gyz + byz, mu)
    byz = byz + (gyz - dyz)
    Lyz = para * back_diff(back_diff(dyz - byz, 1, 0), 1, 2)
    return Lyz, byz

def iter_zz(g, bzz, para, mu):
    """

    :param g: input image
    :param bzz: use to update dxy
    :param para: regularization parameters
    :param mu: parameter which is used to tune fidelity term
    :return: Lzz, bzz
    """
    gzz = back_diff(forward_diff(g, 1, 0), 1, 0)
    dzz = shrink(gzz + bzz, mu)
    bzz = bzz + (gzz - dzz)
    Lzz = para * back_diff(forward_diff(dzz - bzz, 1, 0), 1, 0)
    return Lzz, bzz

def iter_sparse(gsparse,bsparse,para,mu):
    """

    :param gsparse: input image
    :param bzz:  use to update dxy
    :param para: regularization parameters
    :param mu: parameter which is used to tune fidelity term
    :return: Lzz,bzz
    """

    dsparse = shrink(gsparse + bsparse, mu)
    bsparse = bsparse + (gsparse - dsparse)
    Lsparse = para * (dsparse - bsparse)
    return Lsparse, bsparse
