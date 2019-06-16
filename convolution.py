import numpy as np
from PIL import Image
import os
import math


def main():
    """
        Filter implemented:
            - mean
            - median
            - gaussian

        feature
            - kernel is resizable
    """
    kernel_size = 3

    # mean_image = Convolution("images/monnalisaDisturbed.png", kernel_size)
    # mean_image.apply_mean_filter(3)
    # mean_image.save_image('mean')
    #
    # median_image = Convolution("images/monnalisaDisturbed.png", kernel_size)
    # median_image.apply_mean_filter(3)
    # median_image.save_image('median')
    #
    # gaussian_image = Convolution("images/monnalisaDisturbed.png", kernel_size)
    # gaussian_image.apply_gaussian_filter(1)
    # gaussian_image.save_image('gaussian')

    bilateral_image = Convolution("images/monnalisaDisturbed.png", 21)
    bilateral_image.apply_bilateral(100, 100)
    bilateral_image.save_image('bilateral')




class Convolution:

    def __init__(self, img, kernel_size):
        self.kernel_size = kernel_size
        self.image = self.load_image(img)
        self.padimg = self.apply_padding(0)
        self.result = np.ndarray(self.image.size)


    def load_image(self, relative_path):
        """
            Load image given the relative path
            Params:
                - relative_path: path to image relative to root directory
            Return:
                - greyscale image
        """
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, relative_path)
        im = Image.open(filename)
        # check img size
        if len(im.size)>2:
            print("this image is not in greyscale!")
        # convert to numpy array and return
        return np.array(im.convert("L"))


    def apply_padding(self, value):
        """
            Return padded image
            Params:
                - image: image to pad
                - kernel_size: kernel size
                - value: padding dimension
            Return:
                - padded_image: image padded
        """
        image = self.image
        kernel_size = self.kernel_size
        pad = int((kernel_size-1)/2)
        padded_image = np.full((image.shape[0]+(pad*2), image.shape[1]+(pad*2)), value)
        padded_image[pad:-pad, pad:-pad] = image
        return padded_image


    def save_image(self, name):
        """
            Save image in current folder/images
            Params:
                - image: image to save
                - name: name of the image
            Return:
                none
        """
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "images/" + name + ".png")
        im = Image.fromarray(self.result).convert('RGB')
        im.save(filename)


    def apply_median_filter(self, size):
        """
            Apply median filter
            Params:
                - padimg: image padded
                - size: size of the image
            Return:
                - filtered image
        """
        padimg = self.padimag
        pad = int((size - 1) / 2)
        width = padimg.shape[0] - (2 * pad)
        height = padimg.shape[1] - (2 * pad)
        filtered_image = np.ndarray((width, height))

        for i in range(pad, padimg.shape[0] - pad):
            for j in range(pad, padimg.shape[1] - pad):
                filtered_image[i-pad, j-pad] = np.sort(padimg[i - pad:i + pad + 1, j - pad:j + pad + 1].flatten())[int(((size*size)-1)/2)]

        self.result = filtered_image
        return filtered_image


    def generate_gaussian_kernel(self, dim, sigma):
        """
            Generate gaussian kernel
            Params:
                - dim: kernel dimension (it is square)
                - sigma: sigma value
            Return:
                - gaussian kernel
        """
        kernel = np.ndarray((dim, dim))
        center = int((dim-1)/2)

        coeff = 1/((sigma**2)*2*math.pi)
        exp_denum = 2*(sigma**2)

        for i in range(kernel.shape[0]):

            # contribute exponential due to x
            exp_x = (i-center)**2

            for j in range(kernel.shape[1]):

                # contibute to exp due to y
                exp_y = (j-center)**2

                kernel[i, j] = coeff * math.exp(-1*(exp_x + exp_y)/exp_denum)

        return kernel


    def apply_gaussian_filter(self, sigma):
        """
            Apply gaussian filter to a give image
            Params:
                - padimg: padded image
                - kernel_size: size of the kernel
            Return:
                - filtered image
        """
        padimg = self.padimg
        kernel_size = self.kernel_size
        kernel = self.generate_gaussian_kernel(kernel_size, sigma)
        filtered_image = self.apply_filter(kernel)
        self.result = filtered_image
        return filtered_image


    def generate_mean_kernel(self, dim):
        """
            Generate mean kernel
            Params:
                - dim: kernel dimension (it is square)
            Return:
                - mean kernel
        """
        value = 1/(dim**2)
        kernel = np.full((dim, dim), value)
        return kernel


    def apply_mean_filter(self, kernel_size):
        """
            Apply mean filter
            Params:
                - padimg: padded image
                - kernel_size: size of the kernel
            Return:
                - filtered image
        """
        padimg = self.padimg
        kernel = self.generate_mean_kernel(kernel_size)
        filtered_image = self.apply_filter(kernel)
        self.result = filtered_image
        return filtered_image


    def apply_bilateral(self, sigma_d=1, sigma_r=1):

        padimg = self.padimg
        pad = int((self.kernel_size - 1) / 2)
        width = padimg.shape[0] - (2 * pad)
        height = padimg.shape[1] - (2 * pad)
        filtered_image = np.ndarray((width, height))

        for l in range(pad, padimg.shape[0]-pad):
            for m in range(pad, padimg.shape[1]-pad):

                kernel = np.ndarray((self.kernel_size, self.kernel_size))
                norm = 0

                for i in range(self.kernel_size):
                    for j in range(self.kernel_size):
                        # contribute to gaussian of the geometric closeness distance (distance in domain)
                        c_contrib = ( (i-l)**2 + (j-m)**2 ) / sigma_d**2
                        # contribute to gaussian of the similarity function (distance in range values)
                        s_contrib = ((padimg[l-(i-pad), m-(j-pad)] - padimg[l, m]) / sigma_r)**2
                        kernel[i, j] = math.exp((-0.5*(c_contrib+s_contrib)))

                        # normalization
                        norm = norm + kernel[i, j]

                filtered_image[l - pad, m - pad] = np.sum(padimg[l - pad: l + pad + 1, m - pad: m + pad + 1] * kernel) / norm


        self.result = filtered_image


    def apply_filter(self, kernel):
        """"
        This function computes convolution
        Params:
            - image padded: 2x2 matrix
            - image dim (not padded)
            - kernel dimension
        Return:
            - image filtered
        """
        padimg = self.padimg
        pad = int((kernel.shape[0]-1)/2)
        width = padimg.shape[0]-(2*pad)
        height = padimg.shape[1]-(2*pad)
        filtered_image = np.ndarray((width, height))

        for i in range(pad, padimg.shape[0]-pad):
            for j in range(pad, padimg.shape[1]-pad):
                filtered_image[i-pad, j-pad] = np.sum(padimg[i-pad:i+pad+1, j-pad:j+pad+1]*kernel)

        self.result = filtered_image
        return filtered_image


if __name__ == "__main__":
    main()
