import numpy as np
from numpy import ndarray
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


def generate_gaussian_kernel(kernel_size):
    if kernel_size == 1: return np.array([[1]])
    kernel_1d = (np.poly1d([1, 1]) ** (kernel_size - 1)).c
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d.reshape(1, -1)


def expand(image, filter_size):
    kernel_row = generate_gaussian_kernel(filter_size)
    kernel_col = kernel_row.T
    out_shape = (image.shape[0] * 2, image.shape[1] * 2)
    expanded_im = np.zeros(out_shape)
    expanded_im[::2, ::2] = image
    blurred_col = convolve2d(expanded_im, kernel_col * 2, mode="same", boundary="symm")
    blurred_im = convolve2d(blurred_col, kernel_row * 2, mode="same", boundary="symm")
    return blurred_im


def build_single_channel_gaussian_pyramid(im: ndarray, max_levels:int, filter_size:int):
    kernel_1d = generate_gaussian_kernel(filter_size)
    pyr = [im]
    current_im = im
    for _ in range(max_levels - 1):
        blurred_im = convolve2d(convolve2d(current_im, kernel_1d.T, mode="same", boundary="symm"), kernel_1d,
                                mode="same", boundary="symm")
        downsampled_im = blurred_im[::2, ::2]
        if downsampled_im.shape[0] < 2 or downsampled_im.shape[1] < 2: break
        pyr.append(downsampled_im)
        current_im = downsampled_im
    return pyr

def build_gaussian_pyramid(img_path: str, max_levels: int, filter_size: int) -> list:
    """
    Builds a Gaussian pyramid for an RGB image.

    Args:
        img_path (str): Path to the input RGB image.
        max_levels (int): Maximum number of levels in the pyramid.
        filter_size (int): Size of the Gaussian filter (must be odd).

    Returns:
        list: A list containing three lists (one for each channel) of numpy arrays representing the pyramid levels.
    """
    im = plt.imread(img_path)
    if im.ndim == 2: # Grayscale image
        return build_single_channel_gaussian_pyramid(im, max_levels, filter_size)
    ch1_pyr = build_single_channel_gaussian_pyramid(im[:, :, 0], max_levels, filter_size)
    ch2_pyr = build_single_channel_gaussian_pyramid(im[:, :, 1], max_levels, filter_size)
    ch3_pyr = build_single_channel_gaussian_pyramid(im[:, :, 2], max_levels, filter_size)
    united_pyr = unite_pyramid_channels(ch1_pyr, ch2_pyr, ch3_pyr)

    return united_pyr


# --- NEW FUNCTION: Convert levels to 0-255 ---
def convert_pyr_to_uint8(pyr):
    """
    Converts a floating point Laplacian pyramid to uint8 [0, 255].
    - Difference levels are scaled and shifted by 128 (0 becomes 128).
    - The last level (residual) is just scaled (0 becomes 0).
    """
    new_pyr = []
    for i, level in enumerate(pyr):
        # Check if it's the last level (the small residual image)
        is_last_level = (i == len(pyr) - 1)

        if is_last_level:
            # Just scale 0-1 to 0-255
            level_255 = level * 255
        else:
            # Shift 0 to 128, and scale
            # We assume edges are roughly -0.5 to 0.5, so *255 keeps dynamic range
            level_255 = (level * 255) + 128

        # Clip to ensure valid range and cast to uint8
        level_uint8 = np.clip(level_255, 0, 255).astype(np.uint8)
        new_pyr.append(level_uint8)

    return new_pyr


def build_single_channel_laplacian_pyramid(image:ndarray, max_levels:int, filter_size:int):
    gaussian_pyr = build_single_channel_gaussian_pyramid(image, max_levels, filter_size)
    laplacian_pyr = []
    for i in range(len(gaussian_pyr) - 1):
        g_current = gaussian_pyr[i]
        g_next = gaussian_pyr[i + 1]
        expanded_next = expand(g_next, filter_size)

        # Crop if sizes don't match
        if expanded_next.shape != g_current.shape:
            expanded_next = expanded_next[:g_current.shape[0], :g_current.shape[1]]

        laplacian_pyr.append(g_current - expanded_next)

    laplacian_pyr.append(gaussian_pyr[-1])

    # CONVERT TO UINT8 BEFORE RETURNING
    return convert_pyr_to_uint8(laplacian_pyr)


def unite_pyramid_channels(ch1, ch2, ch3):
    united_pyr = []
    for l1, l2, l3 in zip(ch1, ch2, ch3):
        united_pyr.append(np.dstack((l1, l2, l3)))
    return united_pyr


def build_laplacian_pyramid(image: ndarray, max_levels: int, filter_size: int) -> list:
    # Normalize to [0, 1] float
    if image.max() > 1.0 or image.dtype == np.uint8:
        image = image.astype(float) / 255.0

    if image.ndim == 2:
        return build_single_channel_laplacian_pyramid(image, max_levels, filter_size)

    ch1_pyr = build_single_channel_laplacian_pyramid(image[:, :, 0], max_levels, filter_size)
    ch2_pyr = build_single_channel_laplacian_pyramid(image[:, :, 1], max_levels, filter_size)
    ch3_pyr = build_single_channel_laplacian_pyramid(image[:, :, 2], max_levels, filter_size)

    return unite_pyramid_channels(ch1_pyr, ch2_pyr, ch3_pyr)


def image_blending(im1, im2, mask, max_levels, filter_size):
    """
    Blends two RGB images using pyramid blending.

    Args:
        im1, im2: RGB images normalized to [0, 1].
        mask: Grayscale mask normalized to [0, 1]. (1.0 = im1, 0.0 = im2).
    """
    # 1. Build Laplacian pyramids for the RGB images
    # (Assuming build_laplacian_pyramid can accept arrays, see note below)
    L1 = build_laplacian_pyramid(im1, max_levels, filter_size)
    L2 = build_laplacian_pyramid(im2, max_levels, filter_size)

    # 2. Build GAUSSIAN pyramid for the mask
    # The mask must be blurred so the transition is smooth at lower resolutions
    Gm = build_gaussian_pyramid(mask, max_levels, filter_size)

    blended_pyr = []

    # 3. Blend level by level
    for l1, l2, gm in zip(L1, L2, Gm):

        # FIX: The mask is 2D (H, W), but images are 3D (H, W, 3).
        # We need to reshape mask to (H, W, 1) to multiply against RGB.
        if gm.ndim == 2:
            gm = gm[:, :, np.newaxis]

        # The Blending Formula: L_out = (Mask * L1) + ((1 - Mask) * L2)
        blended_level = gm * l1 + (1 - gm) * l2
        blended_pyr.append(blended_level)

    # 4. Collapse the pyramid to get the final image
    im_blended = reconstruct_from_laplacian_pyramid(blended_pyr, filter_size)

    # Clip values to ensure valid image range [0, 1]
    return np.clip(im_blended, 0, 1)

def show_image(img: ndarray) -> None:
    # Now simply display, as the pyramid itself handles the range
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def reconstruct_from_laplacian_pyramid(pyr, filter_size):
    """
    Reconstructs an RGB image from its Laplacian Pyramid.
    We start from the top (smallest) level and expand it, adding the details
    from the current level as we go down.
    """
    # Start with the base (the tiny residual image)
    current_im = pyr[-1]

    # Iterate backwards from the second-to-last level down to 0
    for i in range(len(pyr) - 2, -1, -1):
        # 1. Upsample and blur the current image
        expanded_im = expand(current_im, filter_size)

        # 2. Get the corresponding Laplacian level
        laplacian_level = pyr[i]

        # 3. Handle size mismatch (crop expanded image to match the level)
        # This happens if original dims were odd
        if expanded_im.shape[0] != laplacian_level.shape[0] or \
                expanded_im.shape[1] != laplacian_level.shape[1]:
            expanded_im = expanded_im[:laplacian_level.shape[0], :laplacian_level.shape[1]]

        # 4. Add the details back
        current_im = expanded_im + laplacian_level

    return current_im

if __name__ == "__main__":
    im_path = "im.png"
    im = plt.imread(im_path)
    levels = 5
    filter_size = 5

    # Resulting pyramid is now list of uint8 arrays
    pyramid = build_laplacian_pyramid(im, levels, filter_size)

    for i, level in enumerate(pyramid):
        print(f"Level {i} | Mean: {level.mean():.2f} | Range: [{level.min()}, {level.max()}]")
        show_image(level)