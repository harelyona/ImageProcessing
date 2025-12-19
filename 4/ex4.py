import os
from typing import Tuple, Any, List
import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy import floating, complexfloating, timedelta64
from openpyxl.styles.alignment import horizontal_alignments
from mediapy import read_video
from scipy.signal import convolve2d
from square_video import *

PYRAMID_LEVELS = 5
FILTER_SIZE = 3
ITERATIONS_PER_LEVEL = 10

def show_image(img, save_path=None):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    if save_path:
        plt.imsave(save_path, img, cmap='gray')
    plt.show()

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
    return laplacian_pyr

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


def build_laplacian_pyramid(image, max_levels, filter_size):
    if image.max() > 1.0 or image.dtype == np.uint8:
        image = image.astype(float) / 255.0

    if image.ndim == 2:
        return build_single_channel_laplacian_pyramid(image, max_levels, filter_size)

    ch1_pyr = build_single_channel_laplacian_pyramid(image[:, :, 0], max_levels, filter_size)
    ch2_pyr = build_single_channel_laplacian_pyramid(image[:, :, 1], max_levels, filter_size)
    ch3_pyr = build_single_channel_laplacian_pyramid(image[:, :, 2], max_levels, filter_size)

    return unite_pyramid_channels(ch1_pyr, ch2_pyr, ch3_pyr)


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


def find_rotation_angle(I1: np.ndarray, I2: np.ndarray) -> int:
    """
    Estimates the rotation angle (theta) from I1 to I2.
    Uses a Pyramid and Joint Solver (u, v, theta) to robustly distinguish
    true rotation from large translations.
    """
    # Configuration
    PYRAMID_LEVELS = 3
    FILTER_SIZE = 3
    ITERATIONS = 10

    # Build Pyramids
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
            # J_theta (Calculated) = x*Iy - y*Ix
            # Theoretical Deriv dI/dTh = y*Ix - x*Iy
            # So J_theta here is effectively -dI/dTh (Negative Derivative)
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

            # b Calculation
            # We need b = - (Jacobian * Error)
            # b_u = - sum(-Ix * It) = sum(Ix * It)
            # b_v = - sum(-Iy * It) = sum(Iy * It)
            # b_th = - sum(J_theta * It)  <-- Wait!
            # Since J_theta is ALREADY negative derivative,
            # - (J_theta * It) would be Positive Gradient. We want Negative Gradient.
            # So we strictly want: b_th = sum(J_theta * It)

            b = np.array([
                np.dot(Ix_f, It_f),
                np.dot(Iy_f, It_f),
                np.dot(Jth_f, It_f)  # [CORRECTION] Removed the negative sign here
            ])

            try:
                # Use pinv for stability on featureless images (like the square)
                delta = np.linalg.pinv(A) @ b
                du, dv, dtheta = delta

                # Additive Update
                u += du
                v += dv
                theta += dtheta

                if abs(du) < 1e-3 and abs(dv) < 1e-3 and abs(dtheta) < 1e-3:
                    break
            except np.linalg.LinAlgError:
                break

    # Return only the rotation component
    return -int(round(theta))


def lucas_kanade(frame1: np.ndarray, frame2: np.ndarray) -> Tuple[int, int, int]:
    """
    1. Finds rotation theta using a pyramid (robust to large shifts).
    2. Rotates frame2 by -theta to align with frame1.
    3. Uses lk_for_x_y to find translation (u, v).
    """
    I1 = lk_prep_image(frame1)
    I2 = lk_prep_image(frame2)

    # Step 1: Find Rotation Angle
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
    u, v = lk_for_x_y(I1, I2_aligned)

    return int(u), int(v), int(round(theta))


def create_empty_panorama(video: np.ndarray) -> np.ndarray:
    """
    Computes the size of the panorama needed to fit all frames based on the shifts.
    Returns (min_x, max_x, min_y, max_y).
    """
    frame_hight, frame_width = video.shape[1], video.shape[2]
    xshifts, yshifts, _ = get_video_shifts(video)
    cum_xshifts = np.cumsum(xshifts)
    cum_yshifts = np.cumsum(yshifts)
    panorama_size = frame_hight + cum_xshifts[-1], frame_width + cum_yshifts[-1]
    return np.zeros(panorama_size, dtype=video.dtype)


def get_video_shifts(video: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes (dx, dy, dtheta) between consecutive frames using Lucas-Kanade.
    """
    num_frames = video.shape[0]
    x_shifts = np.zeros(num_frames)
    y_shifts = np.zeros(num_frames)
    th_shifts = np.zeros(num_frames)

    # Calculate shifts for each pair of frames
    for i in range(num_frames - 1):
        x_shifts[i + 1], y_shifts[i + 1], th_shifts[i + 1] = lucas_kanade(video[i], video[i + 1])

    return x_shifts, y_shifts, th_shifts


def get_transform_matrix(dx: float, dy: float, dtheta: float) -> np.ndarray:
    """
    Creates a 3x3 Euclidean transformation matrix.
    """
    # Convert degrees to radians
    theta_rad = np.radians(dtheta)
    c, s = np.cos(theta_rad), np.sin(theta_rad)

    # Standard Euclidean Matrix
    M = np.eye(3)
    M[0, 0] = c
    M[0, 1] = -s
    M[0, 2] = dx
    M[1, 0] = s
    M[1, 1] = c
    M[1, 2] = dy

    return M


def get_centered_cumulative_transforms(dx: np.ndarray, dy: np.ndarray, dtheta: np.ndarray) -> List[np.ndarray]:
    """
    Computes cumulative transforms anchored to the MIDDLE frame.
    """
    num_frames = len(dx)
    center_idx = num_frames // 2

    transforms = [None] * num_frames
    transforms[center_idx] = np.eye(3)

    # Forward Pass (Center -> End)
    current_T = np.eye(3)
    for i in range(center_idx + 1, num_frames):
        # FIX: We use NEGATIVE shifts because LK returns image motion,
        # but we want to place the NEXT frame relative to the CURRENT one.
        # If image moves Left (negative u), Camera moved Right (positive x).
        # Therefore, we subtract the LK shift.
        M_local = get_transform_matrix(-dx[i], -dy[i], -dtheta[i])

        current_T = current_T @ M_local
        transforms[i] = current_T

    # Backward Pass (Center -> Start)
    current_T = np.eye(3)
    for i in range(center_idx, 0, -1):
        # For moving backwards, we normally invert the forward transform.
        # Since Forward = -Shift, Backward = -(-Shift) = +Shift.
        M_local = get_transform_matrix(dx[i], dy[i], dtheta[i])

        current_T = current_T @ M_local
        transforms[i - 1] = current_T

    return transforms


def create_panorama(video: np.ndarray, video_shifts: np.ndarray = None) -> np.ndarray:
    if len(video) == 0:
        return np.zeros((0, 0), dtype=video.dtype)

    h, w = video.shape[1], video.shape[2]

    if video_shifts is not None:
        dx, dy, dtheta = video_shifts
    else:
        dx, dy, dtheta = get_video_shifts(video)

    # 1. Compute Transforms (with corrected signs)
    transforms = get_centered_cumulative_transforms(dx, dy, dtheta)

    # 2. Calculate Canvas Bounds
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ]).T

    all_x, all_y = [], []
    for T in transforms:
        warped_corners = T @ corners
        all_x.extend(warped_corners[0, :])
        all_y.extend(warped_corners[1, :])

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    pano_w = int(np.ceil(max_x - min_x))
    pano_h = int(np.ceil(max_y - min_y))

    # 3. Allocate Canvas
    if video.ndim == 3:
        panorama = np.zeros((pano_h, pano_w), dtype=video.dtype)
    else:
        panorama = np.zeros((pano_h, pano_w, video.shape[3]), dtype=video.dtype)

    # 4. Paste Frames
    T_global_shift = np.eye(3)
    T_global_shift[0, 2] = -min_x
    T_global_shift[1, 2] = -min_y

    for i, frame in enumerate(video):
        T_frame_to_global = transforms[i]
        M_final = T_global_shift @ T_frame_to_global

        warped_frame = cv2.warpAffine(
            frame,
            M_final[:2, :],
            (pano_w, pano_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT
        )

        if warped_frame.ndim == 3:
            mask = np.any(warped_frame > 0, axis=2)
        else:
            mask = (warped_frame > 0)

        panorama[mask] = warped_frame[mask]

    return panorama

boat_x = 0,0.00000,-7.00000,-5.00000,-4.00000,-3.00000,-3.00000,-4.00000,-3.00000,-3.00000,-4.00000,-4.00000,-3.00000,-2.00000,-3.00000,-4.00000,-4.00000,-2.00000,-4.00000,-2.00000,-4.00000,-4.00000,-3.00000,-3.00000,-5.00000,-4.00000,-4.00000,-4.00000,-3.00000,-4.00000,-4.00000,-4.00000,-4.00000,-4.00000,-4.00000,-3.00000,-4.00000,-3.00000,-4.00000,-4.00000,-5.00000,-4.00000,-3.00000,-4.00000,-3.00000,-4.00000,-3.00000,-4.00000,-4.00000,-5.00000,-4.00000,-5.00000,-3.00000,-3.00000,-2.00000,-5.00000,-3.00000,-5.00000,-3.00000,-2.00000,-3.00000,-2.00000,0.00000,-3.00000,-3.00000,-3.00000,-3.00000,-4.00000,-4.00000,-5.00000,-5.00000,-6.00000,-6.00000,-3.00000,-4.00000,-4.00000,-3.00000,-3.00000,-4.00000,-2.00000,-4.00000,-3.00000,-4.00000,-3.00000,-3.00000,-4.00000,-4.00000,-4.00000,-5.00000,-5.00000,-6.00000,-4.00000,-2.00000,-4.00000,-4.00000,-4.00000,-4.00000,-3.00000,-5.00000,-4.00000,-4.00000,-4.00000,-4.00000,-4.00000,-4.00000,-4.00000,-4.00000,-5.00000,-4.00000,-5.00000,-5.00000,-4.00000,-4.00000,-4.00000,-4.00000,-4.00000,-5.00000,-4.00000,-3.00000,-4.00000,-5.00000,-3.00000,-3.00000,-3.00000,-5.00000,-5.00000,-5.00000,-3.00000,-5.00000,-4.00000,-5.00000,-5.00000,-4.00000,-6.00000,-7.00000,-5.00000,-5.00000,-5.00000,-3.00000,-3.00000,-2.00000,-3.00000,-3.00000,-3.00000,-5.00000,-3.00000,-6.00000,-4.00000,-4.00000,-5.00000,-6.00000,-4.00000,-4.00000,-4.00000,-6.00000,-3.00000,-4.00000,-4.00000,-4.00000,-5.00000,-6.00000,-4.00000,-6.00000,-6.00000,-6.00000,-4.00000,-5.00000,-4.00000,-2.00000,-3.00000,-3.00000,-4.00000,-4.00000,-5.00000,-5.00000,-3.00000,-6.00000,-5.00000,-6.00000,-4.00000,-4.00000,-4.00000,-4.00000,-5.00000,-3.00000,-5.00000,-4.00000,-4.00000,-6.00000,-4.00000,-5.00000,-6.00000,-4.00000,-5.00000,-5.00000,-3.00000,-4.00000,-4.00000,-5.00000,-3.00000,-7.00000,-3.00000,-4.00000,-1.00000,-4.00000,-4.00000,-4.00000,-4.00000,-6.00000,-4.00000,-5.00000,-3.00000,-3.00000,-4.00000,-4.00000,-5.00000,-5.00000,-4.00000,-4.00000,-5.00000,-5.00000,-5.00000,-4.00000,-3.00000,-4.00000,-4.00000,-3.00000,-4.00000,-5.00000,-6.00000,-6.00000,-6.00000,-4.00000,-3.00000,-4.00000,-6.00000,-2.00000,-4.00000,-4.00000,-4.00000,-6.00000,-5.00000,-5.00000,-5.00000,-6.00000,-6.00000,-4.00000,-5.00000,-5.00000,-5.00000,-6.00000,-6.00000,-6.00000,-6.00000,-7.00000,-7.00000,-6.00000,-7.00000,-8.00000,-7.00000,-6.00000,-6.00000,-8.00000,-7.00000,-6.00000,-7.00000,-6.00000,-8.00000,-7.00000,-7.00000,-6.00000,-7.00000,-7.00000,-6.00000,-7.00000,-7.00000,-7.00000,-6.00000,-6.00000,-6.00000,-6.00000,-6.00000,-6.00000,-6.00000,-6.00000,-6.00000,-5.00000,-4.00000,-5.00000,-4.00000,-7.00000,-5.00000,-5.00000,-6.00000,-6.00000,-6.00000,-6.00000,-7.00000,-6.00000,-5.00000,-6.00000,-6.00000,-3.00000,-6.00000,-7.00000,-8.00000,-7.00000,-7.00000,-7.00000,-6.00000,-7.00000,-5.00000,-5.00000,-7.00000,-6.00000,-6.00000,-6.00000,-7.00000,-7.00000,-7.00000,-7.00000,-7.00000,-8.00000,-7.00000,-7.00000,-5.00000,-7.00000,-6.00000,-7.00000,-6.00000,-7.00000,-7.00000,-7.00000,-7.00000,-7.00000,-6.00000,-5.00000,-5.00000,-5.00000,-6.00000,-4.00000,-5.00000,-5.00000,-6.00000,-9.00000,-6.00000,-10.00000,-8.00000,-8.00000,-7.00000,-7.00000,-7.00000,-7.00000,-5.00000,-10.00000,-7.00000,-9.00000,-9.00000,-10.00000,-8.00000,-9.00000,-6.00000,-7.00000,-8.00000,-10.00000,-7.00000,-9.00000,-9.00000,-9.00000,-9.00000,-11.00000,-8.00000,-10.00000,-8.00000,-10.00000,-9.00000,-9.00000,-9.00000,-8.00000,-10.00000,-9.00000,-9.00000,-9.00000,-9.00000,-10.00000,-10.00000,-8.00000,-8.00000,-9.00000,-6.00000,-8.00000,-6.00000,-7.00000,-8.00000,-9.00000,-8.00000,-9.00000,-8.00000,-7.00000,-7.00000,-7.00000,-6.00000,-4.00000,-5.00000,-8.00000,-7.00000,-7.00000,-7.00000,-8.00000,-7.00000,-6.00000,-8.00000,-6.00000,-7.00000,-7.00000,-6.00000,-7.00000,-8.00000,-7.00000,-7.00000,-7.00000,-8.00000,-6.00000,-8.00000,-7.00000,-8.00000,-6.00000,-9.00000,-7.00000,-7.00000,-7.00000,-5.00000,-7.00000,-7.00000,-6.00000,-6.00000,-6.00000,-8.00000,-7.00000,-7.00000,-8.00000,-8.00000,-8.00000,-7.00000,-7.00000,-7.00000,-8.00000,-6.00000,-7.00000,-7.00000,-8.00000
boat_y = 1,0.00000,-2.00000,0.00000,0.00000,-3.00000,-2.00000,-1.00000,1.00000,1.00000,0.00000,0.00000,-1.00000,0.00000,0.00000,0.00000,1.00000,0.00000,1.00000,2.00000,0.00000,-1.00000,-1.00000,1.00000,0.00000,0.00000,0.00000,1.00000,1.00000,1.00000,1.00000,2.00000,1.00000,0.00000,-2.00000,0.00000,0.00000,0.00000,0.00000,1.00000,0.00000,0.00000,0.00000,1.00000,1.00000,0.00000,2.00000,2.00000,2.00000,3.00000,1.00000,1.00000,-1.00000,-1.00000,1.00000,0.00000,0.00000,1.00000,2.00000,2.00000,1.00000,0.00000,-2.00000,-1.00000,0.00000,-1.00000,-3.00000,-3.00000,-1.00000,-1.00000,-2.00000,-2.00000,-2.00000,-3.00000,-3.00000,-2.00000,-1.00000,-2.00000,-2.00000,-2.00000,-1.00000,-1.00000,0.00000,1.00000,2.00000,2.00000,1.00000,0.00000,0.00000,1.00000,2.00000,1.00000,0.00000,1.00000,2.00000,1.00000,-1.00000,0.00000,0.00000,-2.00000,-2.00000,0.00000,1.00000,0.00000,0.00000,0.00000,2.00000,1.00000,0.00000,1.00000,0.00000,0.00000,0.00000,-1.00000,-2.00000,-2.00000,0.00000,0.00000,-1.00000,1.00000,2.00000,1.00000,2.00000,1.00000,-1.00000,0.00000,1.00000,0.00000,0.00000,1.00000,1.00000,-1.00000,-1.00000,-1.00000,1.00000,1.00000,0.00000,-2.00000,-2.00000,-1.00000,0.00000,1.00000,1.00000,1.00000,0.00000,-1.00000,0.00000,0.00000,-1.00000,-2.00000,-1.00000,-1.00000,-1.00000,-1.00000,0.00000,1.00000,-1.00000,-2.00000,-1.00000,0.00000,0.00000,-1.00000,-1.00000,1.00000,1.00000,0.00000,-1.00000,1.00000,2.00000,2.00000,1.00000,1.00000,1.00000,3.00000,1.00000,-3.00000,-2.00000,1.00000,1.00000,-1.00000,0.00000,1.00000,1.00000,-1.00000,-1.00000,-1.00000,-1.00000,-2.00000,-1.00000,0.00000,1.00000,1.00000,1.00000,0.00000,0.00000,-1.00000,-2.00000,-2.00000,-2.00000,-1.00000,-2.00000,-2.00000,1.00000,3.00000,2.00000,-1.00000,-1.00000,2.00000,1.00000,-1.00000,1.00000,2.00000,0.00000,-2.00000,-3.00000,-1.00000,0.00000,-2.00000,1.00000,4.00000,2.00000,1.00000,-1.00000,0.00000,0.00000,-1.00000,-5.00000,-4.00000,-2.00000,-3.00000,-4.00000,-2.00000,-1.00000,0.00000,0.00000,1.00000,0.00000,2.00000,1.00000,-1.00000,-1.00000,0.00000,-2.00000,-2.00000,-1.00000,1.00000,0.00000,-1.00000,0.00000,2.00000,2.00000,-1.00000,-1.00000,2.00000,1.00000,-1.00000,-2.00000,2.00000,2.00000,0.00000,1.00000,1.00000,1.00000,0.00000,0.00000,0.00000,-2.00000,-2.00000,-1.00000,-2.00000,-1.00000,1.00000,0.00000,0.00000,0.00000,1.00000,0.00000,1.00000,2.00000,0.00000,0.00000,0.00000,-1.00000,-1.00000,-2.00000,-2.00000,-3.00000,-1.00000,0.00000,1.00000,0.00000,0.00000,1.00000,1.00000,0.00000,0.00000,1.00000,1.00000,1.00000,2.00000,1.00000,0.00000,-1.00000,-1.00000,-1.00000,-3.00000,-2.00000,0.00000,0.00000,-1.00000,-2.00000,-1.00000,1.00000,0.00000,-1.00000,0.00000,0.00000,1.00000,0.00000,1.00000,1.00000,2.00000,0.00000,-1.00000,1.00000,0.00000,-2.00000,-3.00000,-1.00000,0.00000,-1.00000,0.00000,1.00000,1.00000,0.00000,0.00000,0.00000,0.00000,0.00000,-1.00000,-3.00000,0.00000,2.00000,0.00000,0.00000,-1.00000,1.00000,3.00000,2.00000,0.00000,1.00000,2.00000,2.00000,1.00000,0.00000,-3.00000,-4.00000,-2.00000,-1.00000,-2.00000,-3.00000,-1.00000,0.00000,-1.00000,-1.00000,-1.00000,1.00000,0.00000,0.00000,0.00000,-1.00000,0.00000,0.00000,0.00000,-1.00000,-1.00000,-1.00000,0.00000,-1.00000,-1.00000,-1.00000,-1.00000,0.00000,0.00000,0.00000,-1.00000,1.00000,1.00000,0.00000,0.00000,1.00000,1.00000,2.00000,0.00000,-1.00000,0.00000,0.00000,0.00000,1.00000,1.00000,1.00000,1.00000,0.00000,1.00000,1.00000,-1.00000,-2.00000,-1.00000,0.00000,-3.00000,-2.00000,0.00000,1.00000,2.00000,0.00000,-1.00000,1.00000,0.00000,-2.00000,0.00000,1.00000,0.00000,0.00000,1.00000,0.00000,-1.00000,1.00000,1.00000,0.00000,2.00000,1.00000,1.00000,2.00000,3.00000,1.00000,-1.00000,1.00000,1.00000,0.00000,0.00000,2.00000,3.00000,2.00000,-1.00000,0.00000,0.00000,0.00000,-1.00000,1.00000,1.00000,-1.00000
boat_theta = 2,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000
boat_shifts = np.vstack((boat_x, boat_y, boat_theta))
if __name__ == "__main__":
    vid = mp.read_video(r"Exercise Inputs/boat.mp4")
    # panoroma = create_panorama(vid, boat_shifts)
    # show_image(panoroma)
    show_image(vid[0])
