from typing import List
import os
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
INPUTS_DIR = 'inputs'
OUTPUTS_DIR = 'outputs'

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


def create_hybrid_image(im_far, im_close, level,filter_size=5):
    far_pyr = build_gaussian_pyramid(im_far, level + 1, filter_size)
    close_pyr = build_gaussian_pyramid(im_close, level + 1, filter_size)

    # 2. Extract Low Frequencies (Far Image)
    # לוקחים את התמונה הקטנה מהרמה ה-N ומגדילים אותה חזרה לגודל המקורי
    # זה נותן אפקט של Low Pass Filter חזק מאוד
    low_freq_small = far_pyr[level]
    low_freq = resize(low_freq_small, im_far.shape, anti_aliasing=True)

    # 3. Extract High Frequencies (Close Image)
    # לוקחים את הגרסה המטושטשת של התמונה הקרובה
    close_blurred_small = close_pyr[level]
    close_blurred = resize(close_blurred_small, im_close.shape, anti_aliasing=True)

    # מחסירים אותה מהמקור כדי לקבל רק את הפרטים (High Pass)
    high_freq = im_close - close_blurred

    # 4. Combine
    hybrid = low_freq + high_freq

    return np.clip(hybrid, 0, 1)

def show_image(img, save_path=None):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    if save_path:
        plt.imsave(save_path, img, cmap='gray')
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
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    return magnitude_spectrum


def show_fourier_comparison(im1, im2):
    mag_good = get_magnitude_spectrum(im1)
    mag_bad = get_magnitude_spectrum(im2)

    # Calculate a saturation limit (e.g., 95th percentile)
    # This makes the center "burn out" to white, revealing the dark details
    v_max = np.percentile(mag_good, 95)
    v_min = np.min(mag_good)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Good
    axes[0].imshow(mag_good, cmap='gray', vmin=v_min, vmax=v_max)
    axes[0].set_title('Good Blend')

    # Plot Bad
    axes[1].imshow(mag_bad, cmap='gray', vmin=v_min, vmax=v_max)
    axes[1].set_title('Bad Blend')

    # Plot Difference (The Proof)
    # This subtracts the common parts and leaves ONLY the artifact
    diff = np.abs(mag_bad - mag_good)
    axes[2].imshow(diff, cmap='inferno')
    axes[2].set_title('Difference (Bad - Good)')

    for ax in axes: ax.axis('off')
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


def main_blend(im1_path: str, im2_path: str):
    right_img = plt.imread(im1_path)
    left_img = plt.imread(im2_path)
    if left_img.shape != right_img.shape:
        # resize expects (Height, Width, Channels)
        left_img = resize(left_img, right_img.shape, anti_aliasing=True)
    number_of_levels = 8
    filter_size = 5
    mask = np.zeros((right_img.shape[0], right_img.shape[1]))
    mask[:, :mask.shape[1] // 2] = 1.0
    blended_image = image_blending(left_img, right_img, mask,number_of_levels, filter_size)
    show_image(blended_image, )

def main_hybrid():
    far_path = os.path.join(INPUTS_DIR, "dragon.png")
    close_path = os.path.join(INPUTS_DIR, "mouse.png")

    far_image = rgb2gray(plt.imread(far_path))
    close_image = rgb2gray(plt.imread(close_path))

    # כעת אנו משתמשים בפרמטר 'level' במקום 'kernel_size'.
    # נסה לשחק עם level בין 2 ל-5.
    # level=3 בדרך כלל נותן תוצאה מעולה (מקביל לקרנל ענק של ~30-40)
    hybrid_image = create_hybrid_image(far_image, close_image, 3)

    show_image(hybrid_image, os.path.join(OUTPUTS_DIR, "hybrid.png"))

def main_blend_pyramid():
    right_img = plt.imread(os.path.join(INPUTS_DIR, "right.png"))
    left_img = plt.imread(os.path.join(INPUTS_DIR, "left.png"))
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


def main_fft_analysis():
    # 1. Load Source Images ONCE
    right_img = plt.imread(os.path.join(INPUTS_DIR, "right.png"))
    left_img = plt.imread(os.path.join(INPUTS_DIR, "left.png"))

    # 2. Ensure identical size immediately
    if left_img.shape != right_img.shape:
        left_img = resize(left_img, right_img.shape, anti_aliasing=True)

    # 3. Setup Mask
    mask = np.zeros((right_img.shape[0], right_img.shape[1]))
    mask[:, :mask.shape[1] // 2] = 1.0
    filter_size = 5

    # 4. Generate BOTH blends in memory (No saving to disk!)
    # This guarantees they align pixel-perfectly.
    print("Generating Good Blend...")
    good_blend = image_blending(left_img, right_img, mask, max_levels=10, filter_size=filter_size)

    print("Generating Bad Blend...")
    bad_blend = image_blending(left_img, right_img, mask, max_levels=1, filter_size=filter_size)

    # 5. Convert to Grayscale for FFT
    if good_blend.ndim == 3:
        # Handle RGBA or RGB
        if good_blend.shape[-1] == 4: good_blend = good_blend[:, :, :3]
        good_gray = rgb2gray(good_blend)
    else:
        good_gray = good_blend

    if bad_blend.ndim == 3:
        if bad_blend.shape[-1] == 4: bad_blend = bad_blend[:, :, :3]
        bad_gray = rgb2gray(bad_blend)
    else:
        bad_gray = bad_blend

    # 6. Show FFT Comparison
    show_fourier_comparison(good_gray, bad_gray)


if __name__ == "__main__":
    main_hybrid()
