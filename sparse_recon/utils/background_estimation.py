import pywt
import numpy as np
from numpy import zeros


def background_estimation(imgs, th = 1, dlevel = 6, wavename = 'db6', iter = 3):
    img1 = imgs
    ''' Background estimation
        function Background = background_estimation(imgs,th,dlevel,wavename,iter)
        ims: ndarray
            Input image (can be N dimensional).
        th : int, optional
            if iteration {default:1}
        dlevel : int, optional
         decomposition level {default:7}
        wavename
         The selected wavelet function {default:'db6'}
        iter:  int, optional
         iteration {default:3}
        -----------------------------------------------
        Output:
         Background
    '''
    if imgs.ndim < 3:

        [x, y] = imgs.shape

        z = 1
        if x != y:
            if x < y:
                imgs = np.lib.pad(imgs, ((max(x, y) - imgs.shape[0], 0),(0,max(x, y) - imgs.shape[1])), 'symmetric')
            else:
                imgs = np.lib.pad(imgs, ((0,max(x, y) - imgs.shape[0]),(max(x, y) - imgs.shape[1],0)), 'symmetric')
        Background = np.zeros((img1.shape[0], img1.shape[1]), dtype = 'float32')
    else:
        [z, x,y] = imgs.shape
        print(max(x, y) - imgs.shape[1])
        print(max(x, y) - imgs.shape[2])
        if x != y:
            if x < y:
                imgs = np.lib.pad(imgs, ((0,0),(max(x, y) - imgs.shape[1], 0), (0, max(x, y) - imgs.shape[2])), 'symmetric')
            else:
                imgs = np.lib.pad(imgs, ((0,0),(0, max(x, y) - imgs.shape[1]), (max(x, y) - imgs.shape[2], 0)), 'symmetric')
        Background = np.zeros((img1.shape[0],img1.shape[1],img1.shape[2]), dtype = 'float32')
    for frames in range(0,z):
        if imgs.ndim < 3:
            initial = imgs
        else:
            initial = imgs[frames,:,:]
        res = initial
        for ii in range(0,iter):

            m =pywt.wavedec2(res, wavename,'symmetric', dlevel)
            k = pywt.wavedec2(res, wavename, 'symmetric', dlevel)
            #print(k[1][2].shape)
            list_out = []
            for i in k:
                lt = list(i)
                list_out.append(lt)

            n = np.zeros((dlevel + 2, 2))
            for g in range(0,dlevel + 1):
                n[g,:] = np.array(m[g][1].shape)
            n[dlevel + 1,:] = np.array(initial.shape)
            for kk in range(1,dlevel+1):
                list_out[kk][0] = zeros((int(n[kk,1]), int(n[kk,1])), dtype = 'float32')
                list_out[kk][1] = zeros((int(n[kk,1]), int(n[kk,1])), dtype = 'float32')
                list_out[kk][2] = zeros((int(n[kk,1]), int(n[kk,1])), dtype = 'float32')
            Biter = pywt.waverec2(list_out, wavename)

            if th > 0:
                eps = np.sqrt(np.abs(res))/2
                ind = initial > (Biter + eps)
                res[ind] = Biter[ind] + eps[ind]
                k = pywt.wavedec2(res, wavename,'symmetric',dlevel)
                list_out = []
                for i in k:
                    lt = list(i)
                    list_out.append(lt)
                for kk in range(1, dlevel + 1):
                    list_out[kk][0] = zeros((int(n[kk, 1]),int(n[kk, 1])), dtype = 'float32')
                    list_out[kk][1] = zeros((int(n[kk, 1]),int(n[kk, 1])), dtype = 'float32')
                    list_out[kk][2] = zeros((int(n[kk, 1]), int(n[kk, 1])), dtype = 'float32')
                Biter = pywt.waverec2(list_out, wavename)

        if imgs.ndim < 3:
            if x != y:
                if x < y:
                    Background = Biter[y - x:y, 0:y]
                else:
                    Background = Biter[0:x, x - y:x]
        else:
            #Background[frames,:,:] = Biter
            if x != y:
                if x < y:
                    Background[frames, :, :] = Biter[y - x:y, 0:y]
                else:
                    Background[frames, :, :] = Biter[0:x, x - y:x]
    return  Background