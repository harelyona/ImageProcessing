import os
from typing import Tuple, Any
import numpy as np
import cv2
from numpy import floating, complexfloating, timedelta64
from openpyxl.styles.alignment import horizontal_alignments
from mediapy import read_video
from scipy.signal import convolve2d

from square_video import *

import numpy as np



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

def lk_prep_image(im):
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.dtype != np.float32:
        im = im.astype(np.float32)
    # Normalize to 0-1 if in 0-255 range for stability
    if im.max() > 1.0:
        im /= 255.0
    return im


def lk_for_x_y(frame1: np.ndarray, frame2: np.ndarray) -> Tuple[int, int]:
    """
    Computes the global translation (u, v) from frame1 to frame2.
    Returns INTEGER shifts (rounded from sub-pixel accuracy).
    """
    # Configuration
    PYRAMID_LEVELS = 3
    FILTER_SIZE = 3
    ITERATIONS_PER_LEVEL = 10

    I1 = lk_prep_image(frame1)
    I2 = lk_prep_image(frame2)

    pyr1 = build_gaussian_pyramid(I1, PYRAMID_LEVELS, FILTER_SIZE)
    pyr2 = build_gaussian_pyramid(I2, PYRAMID_LEVELS, FILTER_SIZE)

    u, v = 0.0, 0.0

    # Coarse-to-fine iterative refinement
    for level in range(len(pyr1) - 1, -1, -1):
        u *= 2
        v *= 2

        im1_lvl = pyr1[level]
        im2_lvl = pyr2[level]
        h, w = im1_lvl.shape

        for _ in range(ITERATIONS_PER_LEVEL):
            # Warp Image 1 using current estimate
            M = np.float32([[1, 0, -u], [0, 1, -v]])
            im1_warp = cv2.warpAffine(im1_lvl, M, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            # Compute Gradients on the WARPED image
            Ix = cv2.Sobel(im1_warp, cv2.CV_64F, 1, 0, ksize=3)
            Iy = cv2.Sobel(im1_warp, cv2.CV_64F, 0, 1, ksize=3)
            It = im1_warp - im2_lvl

            # Linear System A * d = b
            Ixx = np.sum(Ix * Ix)
            Iyy = np.sum(Iy * Iy)
            Ixy = np.sum(Ix * Iy)
            Ixt = np.sum(Ix * It)
            Iyt = np.sum(Iy * It)

            A = np.array([[Ixx, Ixy], [Ixy, Iyy]])
            b = np.array([[Ixt], [Iyt]])

            try:
                delta = np.linalg.pinv(A) @ b
                du, dv = delta.flatten()
                u += du
                v += dv

                if np.abs(du) < 1e-5 and np.abs(dv) < 1e-5:
                    break
            except np.linalg.LinAlgError:
                break

    # Return rounded integers
    return int(round(u)), int(round(v))


def find_rotation_angle(I1: np.ndarray, I2: np.ndarray) -> float:
    """
    Estimates the rotation angle (theta) from I1 to I2.

    Uses a Pyramid and Joint Solver (u, v, theta) to robustly distinguish
    true rotation from large translations.
    """
    # Configuration
    PYRAMID_LEVELS = 3
    FILTER_SIZE = 3
    ITERATIONS = 10

    # Build Pyramids for robust large-displacement handling
    pyr1 = build_gaussian_pyramid(I1, PYRAMID_LEVELS, FILTER_SIZE)
    pyr2 = build_gaussian_pyramid(I2, PYRAMID_LEVELS, FILTER_SIZE)

    # State: u, v, theta
    u, v, theta = 0.0, 0.0, 0.0

    # Coarse-to-fine loop
    for level in range(len(pyr1) - 1, -1, -1):
        # Scale up translation (theta is scale-invariant)
        u *= 2
        v *= 2

        im1_lvl = pyr1[level]
        im2_lvl = pyr2[level]
        h, w = im1_lvl.shape
        cy, cx = h / 2.0, w / 2.0

        # Grid for rotation derivatives (Centered)
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        x_grid = x_grid.astype(np.float32) - cx
        y_grid = y_grid.astype(np.float32) - cy

        for _ in range(ITERATIONS):
            # 1. Construct Matrix (Rotate then Translate)
            M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
            M[0, 2] -= u
            M[1, 2] -= v

            # 2. Warp I1 towards I2 using Inverse Map
            # This samples I1 at (R^-1 * (x - T))
            im1_warp = cv2.warpAffine(
                im1_lvl, M, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            # 3. Error & Gradients
            It = im1_warp - im2_lvl

            # Gradients on Warped Image
            Ix = cv2.Sobel(im1_warp, cv2.CV_64F, 1, 0, ksize=3) / 8.0
            Iy = cv2.Sobel(im1_warp, cv2.CV_64F, 0, 1, ksize=3) / 8.0

            # 4. Jacobians
            # J_u = -Ix, J_v = -Iy
            # J_theta = (x * Iy - y * Ix)
            J_theta = (x_grid * Iy - y_grid * Ix) * (np.pi / 180.0)

            # 5. Build System
            Ix_f = Ix.flatten()
            Iy_f = Iy.flatten()
            Jth_f = J_theta.flatten()
            It_f = It.flatten()

            Ixx = np.dot(Ix_f, Ix_f)
            Iyy = np.dot(Iy_f, Iy_f)
            Itt = np.dot(Jth_f, Jth_f)
            Ixy = np.dot(Ix_f, Iy_f)
            Ixt = np.dot(Ix_f, Jth_f)
            Iyt = np.dot(Iy_f, Jth_f)

            A = np.array([
                [Ixx, Ixy, Ixt],
                [Ixy, Iyy, Iyt],
                [Ixt, Iyt, Itt]
            ])

            # b = - Sum(Jacobian * Error)
            # b_u = - Sum(-Ix * It) = Sum(Ix * It)
            # b_th = - Sum(J_theta * It)
            b = np.array([
                np.dot(Ix_f, It_f),
                np.dot(Iy_f, It_f),
                -np.dot(Jth_f, It_f)
            ])

            try:
                delta = np.linalg.lstsq(A, b, rcond=None)[0]
                du, dv, dtheta = delta

                # Additive Update (consistent with M construction)
                u += du
                v += dv
                theta += dtheta

                if abs(du) < 1e-3 and abs(dv) < 1e-3 and abs(dtheta) < 1e-3:
                    break
            except np.linalg.LinAlgError:
                break

    # Return only the rotation component
    return theta


def lucas_kanade(frame1: np.ndarray, frame2: np.ndarray) -> Tuple[int, int, int]:
    """
    1. Finds rotation theta (robustly).
    2. Rotates frame2 by -theta to align with frame1.
    3. Uses lk_for_x_y to find translation (u, v).
    """
    I1 = lk_prep_image(frame1)
    I2 = lk_prep_image(frame2)

    # Step 1: Find Rotation Angle
    # Now returns ~0.0 for pure translation inputs
    theta = find_rotation_angle(I1, I2)

    # Step 2: Compensate for Rotation
    h, w = I1.shape
    cx, cy = w / 2.0, h / 2.0

    # Rotate frame2 by -theta to cancel out the detected rotation
    M_fix = cv2.getRotationMatrix2D((cx, cy), -theta, 1.0)

    # Use standard forward warp to apply the fix
    I2_aligned = cv2.warpAffine(
        I2, M_fix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Step 3: Find Translation
    # Now lk_for_x_y receives images that are actually aligned rotationally
    u, v = lk_for_x_y(I1, I2_aligned)

    return int(u), int(v), int(round(theta))


def get_video_shifts(video: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes (dx, dy, dtheta) between consecutive frames.
    """
    num_frames = video.shape[0]
    x_shifts = np.zeros(num_frames)
    y_shifts = np.zeros(num_frames)
    th_shifts = np.zeros(num_frames)

    for i in range(num_frames - 1):
        x_shifts[i], y_shifts[i], th_shifts[i] = lucas_kanade(video[i], video[i + 1])

    return x_shifts, y_shifts, th_shifts

def get_video_shifts(video: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the (x, y) shifts between consecutive frames in a video using Lucas-Kanade optical flow.
    Returns a list of shifts [(dx1, dy1), (dx2, dy2), ...].
    """

    num_frames = video.shape[0]
    x_shifts = np.zeros((num_frames))
    y_shifts = np.zeros((num_frames))
    ang_shifts = np.zeros((num_frames))

    for i in range(num_frames - 1):
        x_shifts[i], y_shifts[i], ang_shifts[i] = lucas_kanade(video[i], video[i + 1])

    return x_shifts, y_shifts, ang_shifts

def create_empty_panorama(video: np.ndarray) -> np.ndarray:
    """
    Computes the size of the panorama needed to fit all frames based on the shifts.
    Returns (min_x, max_x, min_y, max_y).
    """
    frame_hight, frame_width = video.shape[1], video.shape[2]
    xshifts, yshifts = get_video_shifts(video)
    cum_xshifts = np.cumsum(xshifts)
    cum_yshifts = np.cumsum(yshifts)
    panorama_size = frame_hight + cum_xshifts[-1], frame_width + cum_yshifts[-1]
    return np.zeros(panorama_size, dtype=video.dtype)


if __name__ == "__main__":
    # Test with your failing case
    shifts = (-2, 7)
    video = create_diagonal_square_video(100, start_loc=(100, 100), shift=shifts, size=20)

    u, v, theta = lucas_kanade(video[10], video[11])
    print(f"Target: {shifts[0]} {shifts[1]}")
    print(f"Result: {u} {v} {theta}")