import pathlib
from typing import List, Optional, Union, Tuple
import warnings

import imageio
import numpy as np
import os.path as op
from pyrtools import synthetic_images
from skimage import color
import torch
from torch import Tensor

""" The following code is largely based on color_utils in PooledStatisticsMetamers repo. 
The following code is used to convert RGB images to cone LMS and cone opponent color (opc) spaces.
"""

# Define color transformation matrices
rgb2lms = torch.tensor([[0.3811, 0.5783, 0.0402],
                        [0.1967, 0.7244, 0.0782],
                        [0.0241, 0.1288, 0.8444]])
lms2rgb = torch.inverse(rgb2lms)
# A simple approximation of the opponent cone color space (achromatic,red-green,blue-yellow)
lms2opc = torch.tensor([[0.5, 0.5, 0],    # (L+M) / 2)
                       [-4, 4, 0],        # (M-L) * 3
                       [0.5, 0.5, -1]])  # (L+M)/2 - S)

opc2lms = lms2opc.inverse()

# Composite transform from RGB to cone-opponent color space
rgb2opc = torch.matmul(lms2opc,rgb2lms)

opc2rgb = rgb2opc.inverse()

# Short names for the three opponent channels, short for (achromatic,red-green,blue-yellow)
opc_short_names = ('ac','rg','by')


def color_transform_image(image, color_matrix):
    """
    Apply the specified color transformation matrix to an image.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor of shape (C, H, W) or (N, C, H, W) where:
        - N: Number of images in a batch (batch size).
        - C: Number of channels in the image (e.g., 3 for RGB images).
        - H: Height of the image.
        - W: Width of the image.
    color_matrix : torch.Tensor
        Color transformation matrix of shape (3, 3).

    Returns
    -------
    torch.Tensor
        Transformed image tensor.
    """
    if image.dim() == 3:
        return torch.nn.functional.conv2d(image[None, :, :, :], color_matrix[:, :, None, None])[0]
    else:
        return torch.nn.functional.conv2d(image, color_matrix[:, :, None, None])

def rgb_to_coneLMS(image):
    """
    Convert an RGB image to cone LMS space.

    Parameters
    ----------
    image : torch.Tensor
        Input RGB image tensor of shape (C, H, W) or (N, C, H, W).

    Returns
    -------
    torch.Tensor
        Transformed image tensor in cone LMS space.
    """
    return color_transform_image(image, rgb2lms)

def rgb_to_opponentcone(image):
    """
    Convert an RGB image to a cone opponent space image.

    Parameters
    ----------
    image : torch.Tensor
        Input RGB image tensor of shape (C, H, W) or (N, C, H, W).

    Returns
    -------
    torch.Tensor
        Transformed image tensor in cone opponent space.
    """
    return color_transform_image(image, rgb2opc)

def opponentcone_to_rgb(image):
    """
    Convert an RGB image to a cone opponent space image.

    Parameters
    ----------
    image : torch.Tensor
        Input opc image tensor of shape (C, H, W) or (N, C, H, W).

    Returns
    -------
    torch.Tensor
        Transformed image tensor in rgb
    """
    return color_transform_image(image, opc2rgb)

def rescale(image: torch.Tensor, range: tuple = (0, 1)):
    """
    Rescale the pixel values of an image tensor to a specified range.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor of shape (N, C, H, W) where:
        - N: Number of images in a batch (batch size).
        - C: Number of channels in the image (e.g., 3 for RGB images).
        - H: Height of the image.
        - W: Width of the image.
    range : tuple, optional
        The range to rescale the image to, by default (0, 1).

    Returns
    -------
    torch.Tensor
        Rescaled image tensor with pixel values in the specified range.
    torch.Tensor
        Minimum pixel values for each image in the batch, used for inverse rescaling.
    torch.Tensor
        Maximum pixel values for each image in the batch, used for inverse rescaling.

    Notes
    -----
    This function rescales the pixel values of the input image tensor to the specified range
    while preserving the relative differences between pixel values. The min and max values
    are computed across the height and width for each image in the batch.
    """
    # Compute the min and max values across height and width for each image in the batch
    min_val = image.amin(dim=(2, 3), keepdim=True)
    max_val = image.amax(dim=(2, 3), keepdim=True)
    
    # Rescale the images to the range [0, 1]
    image = (image - min_val) / (max_val - min_val + 1e-8)  # Adding epsilon to avoid division by zero
    
    return image, min_val, max_val

def inverse_rescale(image: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor):
    """
    Reverse the rescaling of an image tensor to its original values.

    Parameters
    ----------
    image : torch.Tensor
        Rescaled image tensor of shape (N, C, H, W) where:
        - N: Number of images in a batch (batch size).
        - C: Number of channels in the image (e.g., 3 for RGB images).
        - H: Height of the image.
        - W: Width of the image.
    min_val : torch.Tensor
        Minimum pixel values for each image in the batch, obtained from the `rescale` function.
    max_val : torch.Tensor
        Maximum pixel values for each image in the batch, obtained from the `rescale` function.

    Returns
    -------
    torch.Tensor
        Image tensor with pixel values restored to their original range.

    Notes
    -----
    This function reverses the rescaling applied by the `rescale` function, restoring the original
    pixel values using the min and max values computed during rescaling.
    """
    # Reverse the rescaling to original values
    image = image * (max_val - min_val) + min_val
    return image
