from typing import List

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize


def generate_gaussian_kernel(kernel_size):
    if kernel_size == 1: return np.array([[1]])
    kernel_1d = (np.poly1d([1, 1]) ** (kernel_size - 1)).c
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d.reshape(1, -1)


def expand(image, filter_size):
    """ Upsamples and blurs. Handles 2D and 3D images. """
    kernel_row = generate_gaussian_kernel(filter_size)
    kernel_col = kernel_row.T

    if image.ndim == 3:
        h, w, c = image.shape
        out_shape = (h * 2, w * 2, c)
        expanded_im = np.zeros(out_shape)
        expanded_im[::2, ::2, :] = image
        output_channels = []
        for ch in range(c):
            channel_data = expanded_im[:, :, ch]
            blur_col = convolve2d(channel_data, kernel_col * 2, mode="same", boundary="symm")
            blur_im = convolve2d(blur_col, kernel_row * 2, mode="same", boundary="symm")
            output_channels.append(blur_im)
        return np.dstack(output_channels)
    else:
        out_shape = (image.shape[0] * 2, image.shape[1] * 2)
        expanded_im = np.zeros(out_shape)
        expanded_im[::2, ::2] = image
        blur_col = convolve2d(expanded_im, kernel_col * 2, mode="same", boundary="symm")
        blur_im = convolve2d(blur_col, kernel_row * 2, mode="same", boundary="symm")
        return blur_im


# --- 2. BUILDERS ---

def build_single_channel_gaussian_pyramid(im, max_levels, filter_size):
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


def build_single_channel_laplacian_pyramid(image, max_levels, filter_size):
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

    # FIX: Do NOT convert to uint8 here. Return raw floats.
    return laplacian_pyr


def unite_pyramid_channels(ch1, ch2, ch3):
    united_pyr = []
    for l1, l2, l3 in zip(ch1, ch2, ch3):
        united_pyr.append(np.dstack((l1, l2, l3)))
    return united_pyr


def build_gaussian_pyramid(im, max_levels, filter_size):
    # Normalize if needed
    if im.max() > 1.0 or im.dtype == np.uint8:
        im = im.astype(float) / 255.0

    if im.ndim == 2:
        return build_single_channel_gaussian_pyramid(im, max_levels, filter_size)
    ch1_pyr = build_single_channel_gaussian_pyramid(im[:, :, 0], max_levels, filter_size)
    ch2_pyr = build_single_channel_gaussian_pyramid(im[:, :, 1], max_levels, filter_size)
    ch3_pyr = build_single_channel_gaussian_pyramid(im[:, :, 2], max_levels, filter_size)
    return unite_pyramid_channels(ch1_pyr, ch2_pyr, ch3_pyr)


def build_laplacian_pyramid(image, max_levels, filter_size):
    # Normalize if needed
    if image.max() > 1.0 or image.dtype == np.uint8:
        image = image.astype(float) / 255.0

    if image.ndim == 2:
        return build_single_channel_laplacian_pyramid(image, max_levels, filter_size)

    ch1_pyr = build_single_channel_laplacian_pyramid(image[:, :, 0], max_levels, filter_size)
    ch2_pyr = build_single_channel_laplacian_pyramid(image[:, :, 1], max_levels, filter_size)
    ch3_pyr = build_single_channel_laplacian_pyramid(image[:, :, 2], max_levels, filter_size)

    return unite_pyramid_channels(ch1_pyr, ch2_pyr, ch3_pyr)


# --- 3. BLENDING & RECONSTRUCTION ---

def reconstruct_from_laplacian_pyramid(pyr, filter_size):
    current_im = pyr[-1]
    for i in range(len(pyr) - 2, -1, -1):
        expanded_im = expand(current_im, filter_size)
        laplacian_level = pyr[i]

        if expanded_im.shape[0] != laplacian_level.shape[0] or \
                expanded_im.shape[1] != laplacian_level.shape[1]:
            expanded_im = expanded_im[:laplacian_level.shape[0], :laplacian_level.shape[1]]

        current_im = expanded_im + laplacian_level
    return current_im


def image_blending(im1, im2, mask, max_levels, filter_size):
    # 1. Build Laplacian pyramids (Raw floats)
    L1 = build_laplacian_pyramid(im1, max_levels, filter_size)
    L2 = build_laplacian_pyramid(im2, max_levels, filter_size)

    # 2. Build Gaussian pyramid for mask
    Gm = build_gaussian_pyramid(mask, max_levels, filter_size)

    blended_pyr = []
    for l1, l2, gm in zip(L1, L2, Gm):
        # Handle broadcasting if mask is 2D and images are 3D
        if gm.ndim == 2 and l1.ndim == 3:
            gm = gm[:, :, np.newaxis]

        blended_level = gm * l1 + (1 - gm) * l2
        blended_pyr.append(blended_level)

    im_blended = reconstruct_from_laplacian_pyramid(blended_pyr, filter_size)
    return np.clip(im_blended, 0, 1)


def  create_hybrid_image(im_far, im_close, kernel_size, hybrid_factor=1.5):
    # 1. Low Pass (התמונה הרחוקה - הדרקון)
    kernel_row = generate_gaussian_kernel(kernel_size)
    kernel_col = kernel_row.T

    blur_far = convolve2d(im_far, kernel_col, mode='same', boundary='symm')
    low_freq = convolve2d(blur_far, kernel_row, mode='same', boundary='symm')

    # 2. High Pass (התמונה הקרובה - העכבר)
    blur_close_col = convolve2d(im_close, kernel_col, mode='same', boundary='symm')
    blur_close = convolve2d(blur_close_col, kernel_row, mode='same', boundary='symm')
    high_freq = im_close - blur_close

    # --- התיקון: הגברת התדרים הגבוהים ---
    # נכפיל את הפרטים בפקטור (למשל 1.5 או 2.0) כדי שיהיו ברורים יותר מקרוב
    hybrid = low_freq + high_freq

    return np.clip(hybrid, 0, 1)

def show_image(img, save_path=None):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def add_sub_figure(img, ax):
    ax.imshow(img, cmap='gray')
    ax.axis('off')


def get_magnitude_spectrum(img):
    """
    Computes the log-magnitude spectrum of an image using 2D DFT.
    """
    # 1. Convert to grayscale if needed (Fourier is usually done on intensity)
    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        img = rgb2gray(img)

    # 2. Compute the 2D Fourier Transform
    f = np.fft.fft2(img)

    # 3. Shift the zero-frequency component to the center of the spectrum
    fshift = np.fft.fftshift(f)

    # 4. Compute magnitude and Apply Log Scale
    # We add 1 to avoid log(0) errors. 20 is a standard scaling factor.
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    return magnitude_spectrum


def show_fourier_comparison(good_blend, bad_blend):
    """
    Calculates and plots the Fourier Transform of two images side-by-side.
    """
    # Calculate Spectra
    mag_good = get_magnitude_spectrum(good_blend)
    mag_bad = get_magnitude_spectrum(bad_blend)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Good Blend Spectrum
    axes[0].imshow(mag_good, cmap='gray')
    axes[0].set_title('Magnitude Spectrum (Good Blend)')
    axes[0].axis('off')

    # Bad Blend Spectrum
    axes[1].imshow(mag_bad, cmap='gray')
    axes[1].set_title('Magnitude Spectrum (Bad Blend)')
    axes[1].axis('off')

    plt.show()

def show_pyramid(pyr1: List[np.ndarray], pyr2: List[np.ndarray]):
    """
    Displays pyramid levels side-by-side in large windows.
    """
    for i, (im1, im2) in enumerate(zip(pyr1, pyr2)):
        # 1. Create a NEW figure for each level with a LARGE size
        fig, ax = plt.subplots(1, 2, figsize=(6, 6))

        # 2. Normalize Laplacian levels for display (if needed)
        if im1.min() < 0: im1 = np.clip(im1 + 0.5, 0, 1)
        if im2.min() < 0: im2 = np.clip(im2 + 0.5, 0, 1)

        # 3. Plot
        add_sub_figure(im1, ax=ax[0])
        add_sub_figure(im2, ax=ax[1])
        ax[0].set_title(f"Level {i}")
        ax[1].set_title(f"Level {i}")

        # 5. Show and wait
        plt.show()


def add_brightness(img, value):
    img = img + value
    return np.clip(img, 0, 1)

def main_blend(im1_path: str, im2_path: str):
    right_img = plt.imread(im1_path)
    left_img = plt.imread(im2_path)
    if left_img.shape != right_img.shape:
        # resize expects (Height, Width, Channels)
        left_img = resize(left_img, right_img.shape, anti_aliasing=True)
    number_of_levels = 4
    filter_size = 5
    mask = np.zeros((right_img.shape[0], right_img.shape[1]))
    mask[:, :mask.shape[1] // 2] = 1.0
    blended_image = image_blending(left_img, right_img, mask,number_of_levels, filter_size)
    show_image(blended_image, )
    show_image(mask, )

def main_hybrid(im1_path: str, im2_path: str):
    far_image = rgb2gray(plt.imread(im1_path))
    close_image = rgb2gray(plt.imread(im2_path))

    hybrid_image = create_hybrid_image(far_image, close_image, 64)
    show_image(hybrid_image, "hybrid.png")
    hybrid_image = create_hybrid_image(far_image, close_image, 2)
    show_image(hybrid_image, "bad_hybrid.png")

def main_blend_pyramid():
    right_img = plt.imread("right.png")
    left_img = plt.imread("left.png")
    if left_img.shape != right_img.shape:
        # resize expects (Height, Width, Channels)
        left_img = resize(left_img, right_img.shape, anti_aliasing=True)
    filter_size = 5
    mask = np.zeros((right_img.shape[0], right_img.shape[1]))
    mask[:, :mask.shape[1] // 2] = 1.0
    good_blended_image = image_blending(left_img, right_img, mask, 8, filter_size)
    bad_blended_image = image_blending(left_img, right_img, mask, 2, filter_size)
    # good_pyr = build_gaussian_pyramid(good_blended_image, 10, filter_size)
    # bad_pyr = build_gaussian_pyramid(bad_blended_image, 10, filter_size)
    good_pyr = build_laplacian_pyramid(good_blended_image, 10, filter_size)
    bad_pyr = build_laplacian_pyramid(bad_blended_image, 10, filter_size)
    show_pyramid(good_pyr, bad_pyr)

def main_fft():
    good_img = plt.imread("blended.png")
    bad_img = plt.imread("bad_blended.png")
    if good_img.shape != bad_img.shape:
        # resize expects (Height, Width, Channels)
        good_img = resize(good_img, bad_img.shape, anti_aliasing=True)
    show_fourier_comparison(good_img, bad_img)


if __name__ == "__main__":
    main_fft()

