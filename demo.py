
from sparse_recon.sparse_deconv import sparse_deconv
from skimage import io
from matplotlib import pyplot as plt

if __name__ == '__main__':
    im = io.imread('test.tif')
    plt.imshow(im, cmap = 'gray')
    plt.show()
    pixelsize = 65 #(nm)
    resolution = 280 #(nm)
    img_recon = sparse_deconv(im, resolution / pixelsize)
    plt.imshow(img_recon / img_recon.max() * 255, cmap = 'gray')
    plt.show()
    io.imsave('test_processed.tif', img_recon.astype(im.dtype))