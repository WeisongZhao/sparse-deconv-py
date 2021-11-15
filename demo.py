
from sparse_recon.sparse_deconv import sparse_deconv
from skimage import io
from matplotlib import pyplot as plt

if __name__ == '__main__':
    im = io.imread('test.tif')
    plt.imshow(im, cmap ='gray')
    plt.show()

    img_recon = sparse_deconv(im, [5,5])
    plt.imshow(img_recon / img_recon.max() * 255, cmap ='gray')
    plt.show()