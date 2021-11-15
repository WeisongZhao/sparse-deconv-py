# from sympy import *
import numpy as np
import math
from numpy import log


def Gauss(sigma):
    sigma = np.array(sigma, dtype = 'float32')
    psfN = np.ceil(sigma / math.sqrt(8 * log(2)) * math.sqrt(-2 * log(0.0002))) + 1
    N = psfN * 2 + 1
    sigma = sigma / (2 * math.sqrt(2 * log(2)))
    dim = len(N)
    if dim > 1:
        N[1] = np.maximum(N[0], N[1])
        N[1] = N[0]
        sigma[0: 1] = np.flip(sigma[0: 1])
    if dim == 1:
        x = np.arange(-np.fix(N / 2), np.ceil(N / 2),dtype='float32')
        PSF = np.exp(-0.5 * (x * x) / (np.dot(sigma, sigma)))
        PSF = PSF / PSF.sum()
        center = N / 2 + 1
        return PSF
    if dim == 2:
        m = N[0]
        n = N[1]
        x = np.arange(-np.fix((n / 2)), np.ceil((n / 2)),dtype='float32')
        y = np.arange(-np.fix((m / 2)), np.ceil((m / 2)),dtype='float32')
        X, Y = np.meshgrid(x, y)
        if len(sigma) == 1:
            PSF = np.exp(-X * X / (2 * np.dot(sigma, sigma)) - Y * Y / (2 * np.dot(sigma, sigma)))
            PSFsum = PSF.sum()
        elif len(sigma) == 2:
            s1 = sigma[0]
            s2 = sigma[1]
            PSF = np.exp(-(X * X) / (2 * np.dot(s1, s1)) - (Y * Y) / (2 * np.dot(s2, s2)))
            PSFsum = PSF.sum()
        elif len(sigma) == 3:
            s1 = sigma[0]
            s2 = sigma[1]
            s3 = sigma[2]

            num = -((X * X) * (np.dot(s1, s1)) + ((Y * Y)) * (np.dot(s2, s2)) - 2 * (np.dot(s3, s3)) * ((X * Y)))
            den = 2 * (np.dot(s1, s1) * np.dot(s2, s2) - s3 ** 4)
            PSF = np.exp(num / den)
            PSFsum = PSF.sum()
        PSF = PSF / PSFsum
        center = [m / 2 + 1, n / 2 + 1]
        return PSF
    if dim == 3:
        m = N[0]
        n = N[1]
        k = N[2]
        x = np.arange(-np.fix(n / 2), np.ceil(n / 2),dtype='float32')
        y = np.arange(-np.fix(m / 2), np.ceil(m / 2),dtype='float32')
        z = np.arange(-np.fix(k / 2), np.ceil(k / 2),dtype='float32')
        [X, Y, Z] = np.meshgrid(x, y, z)
        s1 = sigma[0]
        s2 = sigma[1]
        s3 = sigma[2]
        PSF = np.exp(-(X * X) / (2 * s1 * s1) - (Y * Y) / (2 * s2 * s2) - (Z * Z) / (2 * s3 ** 2))
        PSFsum = PSF.sum()
        PSF = PSF / PSFsum
        center = [m / 2 + 1, n / 2 + 1, k / 2 + 1]
        return PSF
'''def Generate_PSF(pixel,lamda,n,NA,z):
    sin2 = ((1 - (1 - math.pow( NA, 2 ))) / 2)
    u = 8 * math.pi * z * sin2 / lamda
    x = np.arange( -n* pixel,n * pixel,pixel)
    X, Y= np.meshgrid(x, x)
    theta2, s1 =cart2pol(X, Y)
    idx = s1 <= 1
    a=X.shape[0]
    b=X.shape[1]
    IP = np.zeros((a,b))
    k = 1
    for  f in range (1,s1.shape[0],1):
        for j in range (1,s1.shape[1],1):
            if idx[f, j] == 0:
                IP[f, j] =0
            else:
                o = s1[idx]
                r = o[k]
                k = k + 1
               # IP[f,j]=lambda p:quad(lambda  x:np.exp(-x),0,np.inf)
                p =symbols('p')
               # besaier =sp.jv(0,np.linspace (0,2* sympy.pi*r* NA/lamda*p,100))
                besaier =scipy.signal.bessel(0,2* sympy.pi*r* NA/lamda*p)
                a= 2*sympy.exp((1j*u*(p**2))/2)
                h=a* besaier
                #h=besaier
                IP[f,j]= integrate(h,(0, 1))
                print(integrate(h, (x, -1, 1)))
    Ipsf=abs(IP**2)
    Ipsf=Ipsf/Ipsf.sum(sum( Ipsf))
    return Ipsf   
'''