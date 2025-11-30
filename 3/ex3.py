import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


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


def create_hybrid_image(im_far, im_close, kernel_size, hybrid_factor=1.5):
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
    hybrid = low_freq + (high_freq * hybrid_factor)

    return np.clip(hybrid, 0, 1)

def show_image(img, save_path=None):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main_blend(im1_path: str, im2_path: str, output_path: str=None):
    right_img = plt.imread(im1_path)
    left_img = plt.imread(im2_path)
    number_of_levels = 10
    filter_size = 5
    mask = np.zeros((right_img.shape[0], right_img.shape[1]))
    mask[:, :mask.shape[1] // 2] = 1.0
    blended_image = image_blending(left_img, right_img, mask, number_of_levels, filter_size)
    show_image(blended_image, output_path)
    show_image(mask, "mask.png")

def main_hybrid(im1_path: str, im2_path: str, output_path: str=None):
    far_image = rgb2gray(plt.imread(im1_path))
    close_image = rgb2gray(plt.imread(im2_path))
    kernel_size = 50
    hybrid_image = create_hybrid_image(far_image, close_image, kernel_size)
    show_image(hybrid_image, output_path)
    show_image(close_image, "close.png")
    show_image(far_image, "far_image.png")


if __name__ == "__main__":
    main_blend("right.png", "left.png", "blended.png")

    main_hybrid("dragon.png", "mouse.png", "hybrid.png")