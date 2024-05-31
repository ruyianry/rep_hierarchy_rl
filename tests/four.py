import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

# Define the input image
image = np.array([
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0]
])
image = np.array([
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0]
])
# image = np.array([
#     [0, 0, 1, 1],
#     [0, 0, 1, 1],
#     [0, 0, 1, 1],
#     [0, 0, 1, 1]
# ])

# Compute the Fourier Transform of the image
fourier_transform = fft2(image)

# Compute the magnitude spectrum
print(image)
ResBlockConv2d
print("Fourier Transform:\n", fourier_transform)


# given the Fourier spectrum, reconstruct the image

spec = [[4,0],[0,2]]
spec = np.array(spec)
print(spec)
print(np.fft.ifft2(spec))
