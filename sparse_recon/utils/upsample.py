import  warnings
import numpy as np

try:
    import cupy as cp
except ImportError:
    cupy = None
xp =np if cp is None else cp
if xp is not cp:
    warnings.warn("could not import cupy... falling back to numpy & cpu.")

def spatial_upsample(SIMmovie,n=2):


    if xp is not cp:
        SIMmovie = SIMmovie
    else:
        SIMmovie = cp.asarray(SIMmovie)

    k = SIMmovie.ndim
    if k > 2:
        [sz,sx,sy] = SIMmovie.shape
        for frames in range(0, sz):
            y = xp.zeros((sz, sx*n,sy*n), dtype = 'float32')
            y = xp.array(y)
            y[frames, 0:sx*n:n, 0:sy*n:n] = SIMmovie[frames,:,:]
            y = xp.array(y)
        return y
    else:
        [sx, sy] = SIMmovie.shape
        y=xp.zeros((sx*n, sy*n), dtype = 'float32')
        y[0:sx * n:n, 0:sy * n:n] = SIMmovie
        return y


def fourier_upsample(imgstack, n = 2):
    ''' 
    Fourier interpolation
    -----------------------------------------------
    imgstack: ndarray
        input image (can be N dimensional).
     n: int, optional
        magnification times  {default: 2}
    ------------------------------------------------
    Output:
      imgfl
    '''
    if xp is not cp:
        imgstack = imgstack
    else:
        imgstack = cp.asarray(imgstack)

    n = n * xp.ones((1, 2))
    if imgstack.ndim < 3:
        z = 1
        sz = [imgstack.shape[0], imgstack.shape[1]]
        imgfl= xp.zeros(( int(n[0][0]) * int(sz[0]), int(n[0][0]) * int(sz[1])))
    else:
        z = imgstack.shape[0]
        sz = [imgstack.shape[1], imgstack.shape[2]]
        imgfl = xp.zeros((z, int(n[0][0]) * int(sz[0]), int(n[0][0]) * int(sz[1])))


    for i in range(0,z):
        if imgstack.ndim < 3:
            img = imgstack
        else:
            img = imgstack[i,:, :]
        imgsz = [img.shape[0], img.shape[1]]
        imgsz = xp.array(imgsz)
        if ((imgsz[0]%2))==1:
            sz = imgsz
        else:
            sz = imgsz-1
        sz = xp.array(sz)
        idx = xp.ceil(sz / 2) + 1 + (n - 1) * xp.floor(sz / 2)
        padsize = [img.shape[0]/2,img.shape[1]/2]
        padsize = xp.array(padsize )
        k = xp.ceil(padsize )
        f=xp.floor(padsize )


       # img = xp.pad(img, [int(k[0]), int(k[1])], 'symmetric')
        img = xp.pad(img, ((int(k[0]), 0), (int(k[1]), 0)), 'symmetric')

        img = xp.pad(img, ((0, int(f[0])), (0, int(f[1]))), 'symmetric')

        im_shape = n*(xp.array(img.shape))
        newsz = xp.floor(im_shape-(n - 1))
        imgl = fInterp_2D(img, newsz)
        if imgstack.ndim < 3:

            imgfl = imgl[int(idx[0][0]):int(n[0][0]) * int(imgsz[0])+int(idx[0][0]), int(idx[0][1]):int(idx[0][1]) + int(n[0][1]) *int(imgsz[1])]
        else:
            imgfl = xp.array(imgfl)

            imgfl[i,:,:] = imgl[int(idx[0][0]):int(n[0][0])*int(imgsz[0])+int(idx[0][0]), int(idx[0][1]):int(idx[0][1])+int(n[0][1])*int(imgsz[1])]

    return imgfl

def fInterp_2D(img, newsz):
    imgsz = img.shape
    imgsz = xp.array(imgsz)
    newsz = xp.array(newsz)
    if (xp.sum(newsz == 0)) >= 1:
        img_ip = []
    isgreater = newsz >= imgsz
    isgreater = isgreater.astype(int)
    isgreater = xp.array(isgreater)
    incr = xp.zeros((2, 1), dtype = 'float32')
    for iDim in range(0,2):
        if isgreater[0][iDim] == 1:
            incr[iDim] = 1
        else:
            incr = xp.floor(imgsz[iDim] / newsz[iDim]) + 1
    newsz[0][0] = int(newsz[0][0])
    a=newsz[0][0]
    b=newsz[0][1]
    nyqst = xp.ceil((imgsz + 1) / 2)
    B = float(a / imgsz[0] * b / imgsz[1])
    img = B * xp.fft.fft2(img)
    img_ip = xp.zeros((int(a), int(b)),dtype = 'complex')
    img_ip[0: int(nyqst[0]), 0: int(nyqst[1])]= img[0: int(nyqst[0]), 0: int(nyqst[1])]#xl, yl
    img_ip[a-(int(imgsz[0])-int(nyqst[0])):a, 0:int(nyqst[1])] = img[int(nyqst[0]):int(imgsz[0]),0:int(nyqst[1])]#xh, yl
    img_ip[0: int(nyqst[0]),a- (int(imgsz[1]) - int(nyqst[1])):a]= img[0: int(nyqst[0]),int( nyqst[1]): int(imgsz[1])]
    img_ip[a-(int(imgsz[0])- int(nyqst[0])):a, a- (int(imgsz[1])-int(nyqst[1])):a]=img[int(nyqst[0]):int(imgsz[0]), int(nyqst[1]):int(imgsz[1])]
    rm = xp.remainder(imgsz, 2)
    if int( rm[0]) == 0 & int(a )!=int(imgsz[0]):
        img_ip[int(nyqst[0]),:] = img_ip[int(nyqst[0]),:] / 2
        img_ip[int(nyqst[0] ) + int(a) - int(imgsz[0]),:] = img_ip[int(nyqst[0]),:]
    if int(rm[1]) == 0 &int(b) != int(imgsz[1]):
        img_ip[ :,int(nyqst[1])] = img_ip[ :,int(nyqst[1])] / 2
        img_ip[:,int(nyqst[1])+int(b)-imgsz[1]] = img_ip[:,int(nyqst[1])]
    img_ip = xp.array(img_ip)
    img_ip =(xp.fft.ifft2(img_ip)). real
    img_ip = img_ip[0: int(a):int(incr[0]), 0:int(b): int(incr[1])]
    # img = cp.float32( img_ip.get())
    # plt.imshow(img , cmap='gray')
    # plt.show()
    return img_ip