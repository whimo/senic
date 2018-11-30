import numpy as np
import cv2
import math
from scipy.signal import convolve2d


def estimate_noise(img):
    '''
    Returns estimated noise level, lower is better. Good images have about 0-2
    '''
    height, width = img.shape

    kernel = [[1, -2, 1],
              [-2, 4, -2],
              [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(img, kernel))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (width - 2) * (height - 2))

    return sigma


def estimate_blur(img):
    '''
    Returns estimated blur level, higher is better. Threshold: about 100
    '''
    return cv2.Laplacian(img, cv2.CV_64F).var()
